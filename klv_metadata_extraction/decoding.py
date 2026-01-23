# decode.py
import jpype
import jpype.imports
from jpype.types import JByte
from utils.logger import get_logger

logger = get_logger(__name__)

class JmisbDecoder:
    def __init__(self, jars):
        self.jars = jars
        self._java_loaded = False

    # ---------------- JVM ----------------
    def start_jvm(self):
        if not jpype.isJVMStarted():
            logger.info("Starting JVM with jars: %s", self.jars)
            jpype.startJVM(classpath=self.jars)
        else:
            logger.debug("JVM already started.")

        # Import Java classes ONLY after JVM starts
        if not self._java_loaded:
            self._load_java_classes()
            self._java_loaded = True
            logger.info("Java classes loaded.")

    def shutdown_jvm(self):
        if jpype.isJVMStarted():
            logger.info("Shutting down JVM.")
            jpype.shutdownJVM()
        else:
            logger.debug("JVM is not running.")

    def _load_java_classes(self):
        # Delayed Java imports (CRITICAL)
        from org.jmisb.api.klv import KlvParser
        from org.jmisb.api.klv.st0903 import VmtiLocalSet, VmtiMetadataKey
        from org.jmisb.api.klv.st0601 import UasDatalinkMessage, UasDatalinkTag, NestedVmtiLocalSet

        self.KlvParser = KlvParser
        self.VmtiLocalSet = VmtiLocalSet
        self.VmtiMetadataKey = VmtiMetadataKey
        self.UasDatalinkMessage = UasDatalinkMessage
        self.NestedVmtiLocalSet = NestedVmtiLocalSet
        self.UasDatalinkTag = UasDatalinkTag

    # ---------------- Helpers ----------------
    @staticmethod
    def safe_value(v):
        return None if v is None else str(v)

    def decode_fields(self, ls):
        out = {}
        for key in ls.getIdentifiers():
            val = ls.getField(key)
            if val:
                out[str(key)] = self.safe_value(val.getDisplayableValue())
        return out

    def decode_vtargets(self, vseries):
        targets = []
        for tgt in vseries.getVTargets():
            tdata = {
                "target_id": int(tgt.getTargetIdentifier()),
                "fields": {}
            }
            for k in tgt.getIdentifiers():
                v = tgt.getField(k)
                if v:
                    tdata["fields"][str(k)] = self.safe_value(v.getDisplayableValue())
            targets.append(tdata)
        return targets

    def decode_algorithms(self, aseries):
        algos = []
        for algo in aseries.getAlgorithms():
            adata = {}
            for k in algo.getIdentifiers():
                v = algo.getField(k)
                if v:
                    adata[str(k)] = self.safe_value(v.getDisplayableValue())
            algos.append(adata)
        return algos

    def decode_ontologies(self, oseries):
        onts = []
        for ont in oseries.getOntologies():
            odata = {}
            for k in ont.getIdentifiers():
                v = ont.getField(k)
                if v:
                    odata[str(k)] = self.safe_value(v.getDisplayableValue())
            onts.append(odata)
        return onts

    # ---------------- Packet Decoders ----------------
    def decode_vmti_packet(self, pkt):
        out = {
            "type": "ST0903_VMTI",
            "fields": self.decode_fields(pkt)
        }

        vseries = pkt.getField(self.VmtiMetadataKey.VTargetSeries)
        if vseries:
            out["vtarget_series"] = self.decode_vtargets(vseries)

        aseries = pkt.getField(self.VmtiMetadataKey.AlgorithmSeries)
        if aseries:
            out["algorithm_series"] = self.decode_algorithms(aseries)

        oseries = pkt.getField(self.VmtiMetadataKey.OntologySeries)
        if oseries:
            out["ontology_series"] = self.decode_ontologies(oseries)

        return out

    def decode_uas_packet(self, pkt):
        out = {
            "type": "ST0601_UAS",
            "fields": self.decode_fields(pkt)
        }

        raw_vmti = pkt.getField(self.UasDatalinkTag.VmtiLocalDataSet)

        if isinstance(raw_vmti, self.NestedVmtiLocalSet):
            vmti = raw_vmti.getVmti()
            vmti_out = {"fields": self.decode_fields(vmti)}

            vseries = vmti.getField(self.VmtiMetadataKey.VTargetSeries)
            if vseries:
                vmti_out["vtarget_series"] = self.decode_vtargets(vseries)

            aseries = vmti.getField(self.VmtiMetadataKey.AlgorithmSeries)
            if aseries:
                vmti_out["algorithm_series"] = self.decode_algorithms(aseries)

            oseries = vmti.getField(self.VmtiMetadataKey.OntologySeries)
            if oseries:
                vmti_out["ontology_series"] = self.decode_ontologies(oseries)

            out["embedded_vmti"] = vmti_out

        return out

    # ---------------- Sync Helper (Updated) ----------------
    def align_to_klv_key(self, byte_data):
        """
        Scans the byte array for the Universal Label (UL)
        that marks the start of a UAS Datalink packet.
        """
        # Ensure we are working with Python bytes (not Java arrays)
        if not isinstance(byte_data, (bytes, bytearray)):
            try:
                # Convert Java byte[] to Python bytes if needed
                byte_data = bytes(byte_data)
            except Exception as e:
                logger.error(f"Failed to convert byte_data to bytes: {e}", exc_info=True)
                return None

        # Standard UAS Datalink Local Set Key (first 8 bytes)
        # 06 0E 2B 34 02 0B 01 01
        uas_key_prefix = b'\x06\x0e\x2b\x34\x02\x0b\x01\x01'
        offset = byte_data.find(uas_key_prefix)
        if offset == -1:
            logger.warning("No valid KLV header found in this segment.")
            return None
        if offset > 0:
            logger.debug(f"Synced: Skipped {offset} bytes to find KLV header.")
            return byte_data[offset:]
        return byte_data

    # ---------------- Main API (Updated) ----------------
    def decode_file(self, klv_path):
        logger.info(f"Decoding KLV file: {klv_path}")
        # 1. Read raw file as standard Python bytes
        # (Do NOT convert to JArray yet)
        try:
            with open(klv_path, "rb") as f:
                raw_bytes = f.read()
        except Exception as e:
            logger.error(f"Failed to read KLV file {klv_path}: {e}", exc_info=True)
            return {"total_packets": 0, "packets": [], "error": str(e)}

        # 2. Sync: Find the start of the KLV stream in Python
        clean_bytes = self.align_to_klv_key(raw_bytes)

        if not clean_bytes:
            logger.warning("No valid KLV header found in this segment.")
            return {"total_packets": 0, "packets": []}

        # 3. Parse: Explicitly cast to Java byte[] here
        # This ensures the parser receives exactly what it expects.
        try:
            j_bytes = jpype.JArray(JByte)(clean_bytes)
            packets = self.KlvParser.parseBytes(j_bytes)
        except Exception as e:
            logger.error(f"Error parsing KLV bytes: {e}", exc_info=True)
            # Optional: Print hex to debug if it fails again
            # logger.debug(f"Failed Header Hex: {clean_bytes[:16].hex()}")
            return {"total_packets": 0, "packets": [], "error": str(e)}

        # 4. Process Results (Your existing logic)
        result = {
            "total_packets": packets.size(),
            "packets": []
        }

        logger.info(f"Decoded {packets.size()} packets from {klv_path}")

        for i in range(packets.size()):
            pkt = packets.get(i)
            pkt_out = {"packet_index": i}

            if isinstance(pkt, self.VmtiLocalSet):
                pkt_out.update(self.decode_vmti_packet(pkt))

            elif isinstance(pkt, self.UasDatalinkMessage):
                pkt_out.update(self.decode_uas_packet(pkt))

            else:
                pkt_out["type"] = "UNKNOWN"
                pkt_out["raw"] = str(pkt)

            result["packets"].append(pkt_out)

        logger.info(f"Finished decoding file: {klv_path}")
        return result