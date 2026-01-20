# decode.py
import jpype
import jpype.imports
from jpype.types import JByte


class JmisbDecoder:
    def __init__(self, jars):
        self.jars = jars
        self._java_loaded = False

    # ---------------- JVM ----------------
    def start_jvm(self):
        if not jpype.isJVMStarted():
            jpype.startJVM(classpath=self.jars)

        # Import Java classes ONLY after JVM starts
        if not self._java_loaded:
            self._load_java_classes()
            self._java_loaded = True

    def shutdown_jvm(self):
        if jpype.isJVMStarted():
            jpype.shutdownJVM()

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

    # ---------------- Main API ----------------
    def decode_file(self, klv_path):
        raw = open(klv_path, "rb").read()
        byte_array = jpype.JArray(JByte)(raw)
        packets = self.KlvParser.parseBytes(byte_array)
        result = {
            "total_packets": packets.size(),
            "packets": []
        }

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

        return result