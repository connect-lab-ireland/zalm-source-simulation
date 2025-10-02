# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# File: zalm_protocols.py
# Jerry Horgan, 2025
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import random
import numpy as np
import netsquid as ns
import pandas as pd
"""
from netsquid.qubits import create_qubits, discard, operate, measure, set_qstate_formalism, QFormalism

from netsquid.qubits import operators as ops
from netsquid.nodes import Node, Connection, Network
from netsquid.components import instructions as instr
from netsquid.components import QuantumMemory
from netsquid.components.qchannel import QuantumChannel
from netsquid.components.cchannel import ClassicalChannel
from netsquid.components.qprogram import QuantumProgram
from netsquid.components.qprocessor import QuantumProcessor, PhysicalInstruction
from netsquid.components import instructions as instr
from netsquid.components.models.qerrormodels import DepolarNoiseModel, DephaseNoiseModel
"""
from netsquid.qubits import ketstates as ks
from netsquid.qubits import qubitapi as qapi
from netsquid.qubits.ketutil import ket2dm
from netsquid.protocols.nodeprotocols import NodeProtocol
from tqdm import tqdm
#from utils import Logg, discard_qubit, calculate_spectral_visibility, calculate_temporal_visibility, find_nearest_channel, determineBellState
from utils import setup_logging, calculate_visibility, determineBellState, find_channel_from_freq, find_freq_from_channel, serialize_qstate
from utils import find_channel_from_freq as find_nearest_channel
from zalm_programs import SPDCInitQubits, SPDCProgram, BellMeasurementProgram, CNOTQubits, PostProcessProgram, PhaseFlipQubits

import config
import warnings

#log = Logg()
log = setup_logging()

H = "H" # = 0
V = "V" # = 1

# =============================================================================
#  DEFINE THE IDEAL TARGET STATE AS A DENSITY MATRIX
# =============================================================================
# ks.b01 is the pure state ket |Ψ+>
# We convert it to a density matrix ρ = |Ψ+><Ψ+|
# This only needs to be done once.
TARGET_PSIplus_DM = ket2dm(ks.b01)
TARGET_PSIminus_DM = ket2dm(ks.b11)

class SignalProtocol(NodeProtocol):
    """
    Protocol to capture the signals from all the other protocols.

    Parameters
    ----------
    None

    Returns
    -------
    Passes signals and their outputs
    """
    def __init__(self, node, name=None):
        super().__init__(node=node, name=name)
        self._trigger_signal = "trigger"
        self._complete_signal = "complete"
        self._bell_state_signal ="bellState"
        self._reset_signal = "reset"
        self.add_signal(self._reset_signal)
        self.add_signal(self._bell_state_signal)
        self.add_signal(self._trigger_signal)
        self.add_signal(self._complete_signal)
        self.ZALM_protocol = None
        self.Heralding_protocol = None
        self.BeamSplitter_protocol = None
        self.MeasureStation_protocol = None
        self.FilterAH_protocol = None
        self.FilterAV_protocol = None
        self.FilterBH_protocol = None
        self.FilterBV_protocol = None
        self.PostPA_protocol = None
        self.PostPB_protocol = None
        self.SPDCa_protocol = None
        self.SPDCb_protocol = None
        self.PBSa_protocol = None
        self.PBSb_protocol = None

    def connect(self, zalm_protocol, spdca_protocol, spdcb_protocol, bs_protocol, PBSa_protocol, PBSb_protocol, measure_protocol, hs_protocol, filterAH_protocol, filterAV_protocol, filterBH_protocol, filterBV_protocol, postA_protocol, postB_protocol):
        """Wire up the connections to other protocols."""
        #print(f"{self.name}: Connecting to other protocols.")
        self.ZALM_protocol = zalm_protocol
        self.Heralding_protocol = hs_protocol
        self.BeamSplitter_protocol = bs_protocol
        self.MeasureStation_protocol = measure_protocol
        self.FilterAH_protocol = filterAH_protocol
        self.FilterAV_protocol = filterAV_protocol
        self.FilterBH_protocol = filterBH_protocol
        self.FilterBV_protocol = filterBV_protocol
        self.PostPA_protocol = postA_protocol
        self.PostPB_protocol = postB_protocol
        self.SPDCa_protocol = spdca_protocol
        self.SPDCb_protocol = spdcb_protocol
        self.PBSa_protocol = PBSa_protocol
        self.PBSb_protocol = PBSb_protocol

    def disconnect(self):
        self.ZALM_protocol = None
        self.Heralding_protocol = None
        self.BeamSplitter_protocol = None
        self.MeasureStation_protocol = None
        self.FilterAH_protocol = None
        self.FilterAV_protocol = None
        self.FilterBH_protocol = None
        self.FilterBV_protocol = None
        self.PostPA_protocol = None
        self.PostPB_protocol = None
        self.SPDCa_protocol = None
        self.SPDCb_protocol = None
        self.PBSa_protocol = None
        self.PBSb_protocol = None

    def run(self):
        log.debug(f"###   {self.node.name} restarting   ###")
        results_list = []
        yield self.await_signal(sender=self.ZALM_protocol, signal_label=self._trigger_signal)
        # Events to listen for
        evtSPDCaError  = self.await_signal(sender=self.SPDCa_protocol, signal_label=self._reset_signal)
        evtSPDCbError  = self.await_signal(sender=self.SPDCb_protocol, signal_label=self._reset_signal)
        evtBSError  = self.await_signal(sender=self.BeamSplitter_protocol, signal_label=self._reset_signal)
        evtPBSaError = self.await_signal(sender=self.PBSa_protocol, signal_label=self._reset_signal)
        evtPBSbError = self.await_signal(sender=self.PBSb_protocol, signal_label=self._reset_signal)
        evtFahError  = self.await_signal(sender=self.FilterAH_protocol, signal_label=self._reset_signal)
        evtFavError  = self.await_signal(sender=self.FilterAV_protocol, signal_label=self._reset_signal)
        evtFbhError  = self.await_signal(sender=self.FilterBH_protocol, signal_label=self._reset_signal)
        evtFbvError  = self.await_signal(sender=self.FilterBV_protocol, signal_label=self._reset_signal)
        evtHRLDError  = self.await_signal(sender=self.Heralding_protocol, signal_label=self._reset_signal)
        evtBSM = self.await_signal(sender=self.Heralding_protocol, signal_label=self._bell_state_signal)

        events = yield (evtSPDCaError | evtSPDCbError | evtBSError | evtPBSaError | evtPBSbError | evtFahError | evtFavError| evtFbhError | evtFbvError | evtHRLDError) | evtBSM
        event = events.triggered_events[-1].source
        #if event.name == "HeraldingProtocol":
        if events.second_term.value:
            BSresult = event.get_signal_result(self._bell_state_signal)
            run_data = {'bsm_outcome': BSresult}

            evtMeasured = self.await_signal(sender=self.MeasureStation_protocol, signal_label=self._complete_signal)
            status = yield evtMeasured
            protocol = status.triggered_events[-1].source
            result = protocol.get_signal_result(self._complete_signal)
            if isinstance(result, dict):
                run_data.update(result)
            else:
                run_data['status'] = result
            results_list.append(run_data)

        else:
            log.warning(f"Signal came from {event.name}")
            run_data = {'bsm_outcome': "UNKNOWN"}
            ERresult = event.get_signal_result(self._reset_signal)
            if isinstance(ERresult, dict):
                run_data.update(ERresult)
            else:
                run_data['status'] = ERresult
            result_payload = { 'status': "ERROR", 'fidelity': 0.0}
            run_data.update(result_payload)

            results_list.append(run_data)
            log.warning(results_list[0])
            #self.send_signal(self._reset_signal)

        # Send 'complete' signal to ZALM with result data
        self.send_signal(self._complete_signal, result=results_list)

class ZALMProtocol(NodeProtocol):
    """
    Runs the NetSquid simulation multiple times and collects the results.

    Parameters
    ----------
    num_runs : int
        The number of times to run the simulation.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the results of all runs.
    """
    def __init__(self, node, name=None):
        super().__init__(node=node, name=name)
        self._trigger_signal = "trigger"
        self._complete_signal = "complete"
        self._bell_state_signal ="bellState"
        self.add_signal(self._bell_state_signal)
        self.add_signal(self._trigger_signal)
        self.add_signal(self._complete_signal)
        self.SignalProtocol = None
        self.result = None

    def connect(self, signal_protocol):
        """Wire up the connections to other protocols."""
        #print(f"{self.name}: Connecting to other protocols.")
        self.SignalProtocol = signal_protocol

    def disconnect(self):
        self.SignalProtocol = None
        self.result = None

    def run(self):
        log.debug(f"###   {self.node.name} restarting   ###")
        self.send_signal(self._trigger_signal)
        status = yield self.await_signal(sender=self.SignalProtocol, signal_label=self._complete_signal)
        protocol = status.triggered_events[-1].source
        result = protocol.get_signal_result(self._complete_signal)

        self.result = result[0]

        return result[0]


class SPDCProcessProtocol(NodeProtocol):

    def __init__(self, node, name=None):
        super().__init__(node=node, name=name)
        self._qubit_sent_signal = "qubit_sent"
        self._qubit_recv_signal = "qubit_recv"
        self._qubit_rts_signal = "qubit_rts"
        self._qubit_cts_signal = "qubit_cts"
        self._qubit_ent_signal = "qubit_ent"
        self._qubit_nent_signal = "qubit_nent"
        self._qubit_lost_signal = "qubit_lost"
        self._postp_rts_signal = "postp_rts"
        self._postp_cts_signal = "postp_cts"
        self._trigger_signal = "trigger"
        self._complete_signal = "complete"
        self._reset_signal = "reset"
        self.add_signal(self._reset_signal)
        self.add_signal(self._complete_signal)
        self.add_signal(self._trigger_signal)
        self.add_signal(self._qubit_lost_signal)
        self.add_signal(self._qubit_ent_signal)
        self.add_signal(self._qubit_nent_signal)
        self.add_signal(self._qubit_sent_signal)
        self.add_signal(self._qubit_recv_signal)
        self.add_signal(self._qubit_rts_signal)
        self.add_signal(self._qubit_cts_signal)
        self.add_signal(self._postp_rts_signal)
        self.add_signal(self._postp_cts_signal)
        self.ZALM_protocol = None
        self.BeamSplitter_protocol = None
        self.PostProcessor_protocol = None

        # --- Pre-calculate constants for this source ---
        # Pump properties
        self.pump_wl_m = config.SPDC_PUMP_WAVELENGTH_NM * 1e-9
        self.pump_freq_hz = 299792458.0 / self.pump_wl_m
        self.centre_freq_hz = self.pump_freq_hz / 2

        # Degeneracy bandwidth (the range of possible center frequencies)
        center_wl_m = self.pump_wl_m * 2
        bandwidth_nm = config.SPDC_DEGENERACY_BANDWIDTH_FWHM_NM
        if bandwidth_nm > 0:
            bandwidth_fwhm_hz = (299792458.0 / (center_wl_m**2)) * (bandwidth_nm * 1e-9)
            self.bandwidth_stdev_hz = bandwidth_fwhm_hz / (2 * np.sqrt(2 * np.log(2)))
        else:
            self.bandwidth_stdev_hz = 0

    def connect(self, zalm_protocol, bs_protocol, post_protocol):
        """Wire up the connections to other protocols."""
        #print(f"{self.name}: Connecting to other protocols.")
        self.ZALM_protocol = zalm_protocol
        self.BeamSplitter_protocol = bs_protocol
        self.PostProcessor_protocol = post_protocol

    def disconnect(self):
        self.ZALM_protocol = None
        self.BeamSplitter_protocol = None
        self.PostProcessor_protocol = None

    def run(self):
        log.debug(f"###   {self.node.name} restarting   ###")
        yield self.await_signal(sender=self.ZALM_protocol, signal_label=self._trigger_signal)
        spdc_init = SPDCInitQubits()
        spdc_program = SPDCProgram()
        yield self.node.qmemory.execute_program(spdc_init)
        q1, q2 = self.node.qmemory.peek(positions=[0, 1], skip_noise=True)
        polarisations = self._addFreqandPolar(q1, q2)
        loss_prob = 1 - config.EMISSION_SUCCESS_PROBABILITY
        if np.random.rand() < loss_prob: #
            self.node.qmemory.pop(positions=[0, 1])
            log.warning(f"{self.node.name}: Emission Loss")
            result_payload = { 'status': "ERROR", 'error_type': "EMMISSION Loss", 'location': self.node.name}
            self.send_signal(self._reset_signal, result=result_payload)
            return False
        yield self.node.qmemory.execute_program(spdc_program, vert_index=polarisations[1], horz_index=polarisations[0])
        q1, q2 = self.node.qmemory.peek(positions=[0, 1], skip_noise=True)
        log.debug("Fidelity of generated entanglement: {}".format(round(ns.qubits.fidelity([q1, q2], ks.b01))))
        log.debug(q1.qstate.qrepr)
        log.debug(q2.qstate.qrepr)
        yield from self._send_signal_to_postproccessor()
        yield from self._send_idler_to_beamsplitter()


    def _addFreqandPolar(self, s, i):
        # Offset centred on 0 => can be + or -
        freq_offset_hz = np.random.normal(0, self.bandwidth_stdev_hz)
        target_freq_hz = self.pump_freq_hz / 2

        # In symmetric mode, the center of the pair is half the pump frequency.
        # In dichroic mode, we could model it as being tuned to a specific channel.
        if config.SPDC_MODE == 'SYMMETRIC' or config.SPDC_MODE == 'PBS':
            idler_freq = self.centre_freq_hz + freq_offset_hz
            signal_freq = self.pump_freq_hz - idler_freq
        else: #SPDC_MODE = 'DICHROIC'
            idler_freq = self.centre_freq_hz - abs(freq_offset_hz)
            signal_freq = self.pump_freq_hz - idler_freq
        s.wavelength = signal_freq
        i.wavelength = idler_freq
        log.debug(f"Signal is {signal_freq} : Idler is {idler_freq}")

        # Set polarisation - and return mem position of Vertical
        if config.SPDC_MODE == 'PBS': # Static - Determined positions for PBS
            if self.name == "a":
                s.polarisation = H
                i.polarisation = V
                ret_value = [0,1]
            elif self.name == "b":
                s.polarisation = V
                i.polarisation = H
                ret_value = [1,0]
        else: # Random allocation for DICHROIC or STANDARD (Beam Splitter)
            set_horizontal = random.choice([s ,i])
            if set_horizontal.name == s.name:
                s.polarisation = H
                i.polarisation = V
                ret_value = [0,1]
            else:
                s.polarisation = V
                i.polarisation = H
                ret_value = [1,0]
        return ret_value

    def _send_idler_to_beamsplitter(self):
        idler = 1
        qubit = self.node.qmemory.peek(positions=[idler], skip_noise=True)
        log.debug(f"{self.node.name}: Peeked qubit is {qubit[0].wavelength}")
        self.send_signal(self._qubit_rts_signal)
        yield self.await_signal(sender=self.BeamSplitter_protocol, signal_label=self._qubit_cts_signal)
        log.debug(f"{self.node.name}: Got a CTS signal - popping")
        res = self.node.qmemory.pop(idler)
        log.info(f"{self.node.name}: Result: {res[0].polarisation}")
        self.node.ports["idler"].tx_output(res[0])
        log.debug(f"{self.node.name}: Sent Qubit")
        yield self.await_signal(sender=self.BeamSplitter_protocol, signal_label=self._qubit_recv_signal)
        self.send_signal(self._qubit_sent_signal)
        log.debug(f"{self.node.name}: Sent SENT Signal")
        evt_ent = self.await_signal(sender=self.BeamSplitter_protocol, signal_label=self._qubit_ent_signal)
        evt_nent = self.await_signal(sender=self.BeamSplitter_protocol, signal_label=self._qubit_nent_signal)
        evt_expr = yield evt_ent | evt_nent
        if evt_expr.first_term.value:
            log.debug(f"{self.node.name}: Received Entanglement signal")
        else:
            log.debug(f"{self.node.name}: Received NON-Entanglement signal")

    def _send_signal_to_postproccessor(self):
        signal = 0
        qubit = self.node.qmemory.peek(positions=[signal], skip_noise=True)
        log.debug(f"{self.node.name}: Peeked Signal qubit is {qubit[0].wavelength}")
        self.send_signal(self._postp_rts_signal)
        yield self.await_signal(sender=self.PostProcessor_protocol, signal_label=self._postp_cts_signal)
        log.debug(f"{self.node.name}: Got a CTS signal - popping")
        res = self.node.qmemory.pop(signal)
        log.info(f"{self.node.name}: Signal is {res[0].polarisation} Polarisation")
        self.node.ports["signal"].tx_output(res[0])
        log.debug(f"{self.node.name}: Sent Qubit")
        yield self.await_signal(sender=self.PostProcessor_protocol, signal_label=self._qubit_recv_signal)
        self.send_signal(self._qubit_sent_signal)
        log.debug(f"{self.node.name}: Sent SENT Signal")

class PBSProtocol(NodeProtocol):

    def __init__(self, node, name=None):
        super().__init__(node=node, name=name)
        self._qubit_recv_signal = "qubit_recv"
        #self._qubit_sent_signal = "qubit_sent"
        self._qubit_rts_signal = "qubit_rts"
        self._qubit_cts_signal = "qubit_cts"
        self._filter_rts_signal = "filter_rts"
        self._filter_cts_signal = "filter_cts"
        self._qubit_ent_signal = "qubit_ent"
        self._qubit_nent_signal = "qubit_nent"
        self._qubit_lost_signal = "qubit_lost"
        self._pbs_rts_signal = "pbs_rts"
        self._pbs_cts_signal = "pbs_cts"
        self._pbs_rts2_signal = "pbs_rts2"
        self._pbs_cts2_signal = "pbs_cts2"
        self._qubit_lost_signal = "qubit_lost"
        self._complete_signal = "complete"
        self._filter_rts2_signal = "filter_rts2"
        self._reset_signal = "reset"
        self.add_signal(self._reset_signal)
        self.add_signal(self._filter_rts2_signal)
        self.add_signal(self._complete_signal)
        self.add_signal(self._qubit_lost_signal)
        self.add_signal(self._pbs_rts_signal)
        self.add_signal(self._pbs_cts_signal)
        self.add_signal(self._qubit_ent_signal)
        self.add_signal(self._qubit_nent_signal)
        self.add_signal(self._qubit_recv_signal)
        #self.add_signal(self._qubit_sent_signal)
        self.add_signal(self._qubit_rts_signal)
        self.add_signal(self._qubit_cts_signal)
        self.add_signal(self._filter_rts_signal)
        self.add_signal(self._filter_cts_signal)
        self.add_signal(self._pbs_rts2_signal)
        self.add_signal(self._pbs_cts2_signal)
        self.BeamSplitter_protocol = None
        self.FilterH_protocol = None
        self.FilterV_protocol = None
        self.SignalProtocol = None
        #self.SignalProtocol = S_protocol = None


    def connect(self, bs_protocol, filterH_protocol, filterV_protocol, S_protocol):
        """Wire up the connections to other protocols."""
        #print(f"{self.name}: Connecting to other protocols.")
        self.BeamSplitter_protocol = bs_protocol
        self.FilterH_protocol = filterH_protocol
        self.FilterV_protocol = filterV_protocol
        self.SignalProtocol = S_protocol
        #self.SignalProtocol = S_protocol = S_protocol

    def disconnect(self):
        self.BeamSplitter_protocol = None
        self.FilterH_protocol = None
        self.FilterV_protocol = None
        self.SignalProtocol = None

    def run(self):

        log.debug(f"###   {self.node.name} restarting   ###")
        qport = self.node.ports["qin"]
        evtRTS1 = self.await_signal(sender=self.BeamSplitter_protocol, signal_label=self._pbs_rts_signal)
        evtRTS2 = self.await_signal(sender=self.BeamSplitter_protocol, signal_label=self._pbs_rts2_signal)
        evntExpr = yield evtRTS1 | evtRTS2
        if evntExpr.first_term.value:
            log.debug(f"{self.node.name}: Received an RTS from the BeamSplitter ")
            self.send_signal(self._pbs_cts_signal)
        else:
            log.debug(f"{self.node.name}: Received an RTS2 from the BeamSplitter ")
            self.send_signal(self._pbs_cts2_signal)
        evt_expr = yield self.await_port_input(qport) | self.await_signal(sender=self.SignalProtocol, signal_label=self._complete_signal)
        if evt_expr.second_term.value:
            return False
        qubitBS = qport.rx_input().items
        log.debug(f"{self.node.name}: Received a qubit from the BeamSplitter {qubitBS}")
        log.debug(f"{self.node.name}: {len(qubitBS)} qubits received")
        pos=0
        for item in qubitBS:
            loss_prob = 1 - 10**(-config.PBS_INSERTION_LOSS_DB / 10)
            if np.random.rand() < loss_prob: # per input qubit
                qapi.discard(qubitBS[pos])
                log.warning(f"{self.node.name}: Insertion Loss")
                result_payload = { 'status': "ERROR", 'error_type': "INSERTION Loss", 'location': self.node.name}
                self.send_signal(self._reset_signal, result=result_payload)
                return False
            try:
                self.node.qmemory.put(qubitBS[pos], positions=pos)
            except:
                log.warning(f"{self.node.name}: Insertion Loss : Qubit failed to put")
                result_payload = { 'status': "ERROR", 'error_type': "INSERTION Loss", 'location': self.node.name}
                self.send_signal(self._reset_signal, result=result_payload)
                return False
            qubit = self.node.qmemory.peek(positions=pos, skip_noise=True)
            log.debug(f"{self.node.name}: Peeked qubit is {qubit[0].wavelength}")
            pos += 1
        self.send_signal(self._qubit_recv_signal)

        pos=0
        outPort = []
        qubit = []
        for item in qubitBS:
            try:
                res = self.node.qmemory.pop(pos)
            except:
                log.warning(f"{self.node.name}: Emission Loss : Qubit failed to pop")
                result_payload = { 'status': "ERROR", 'error_type': "EMISSION LOSS", 'location': self.node.name}
                self.send_signal(self._reset_signal, result=result_payload)
                if pos == 1:
                    qapi.discard(qubitBS[pos-1])
                    #Error = True (empty slot 0 if 1)
                return False
            log.info(f"{self.node.name}: Result: {res[0].polarisation}")

            E_ratio = config.PBS_EXTINCTION_RATIO
            if res[0].polarisation == "H":
                if random.random() < E_ratio:
                    outPort.append("qoutV")
                    log.warning(f"{self.node.name}: Extinction Error")
                    result_payload = { 'status': "ERROR", 'error_type': "Extinction",'location': self.node.name}
                    self.send_signal(self._reset_signal, result=result_payload)
                    if pos == 1:
                        qapi.discard(qubitBS[pos-1])
                    return False
                else:
                    outPort.append("qoutH")
            else:
                if random.random() < E_ratio:
                    outPort.append("qoutH")
                    log.warning(f"{self.node.name}: Extinction Error")
                    result_payload = { 'status': "ERROR", 'error_type': "Extinction",'location': self.node.name}
                    self.send_signal(self._reset_signal, result=result_payload)
                    if pos == 1:
                        qapi.discard(qubitBS[pos-1])
                    return False
                else:
                    outPort.append("qoutV")
            pos += 1
            qubit.append(res[0])

        if len(outPort) == 1:
            self.send_signal(self._filter_rts_signal)
        elif len(outPort) == 2:
            if outPort[0] == outPort[1]:
                self.send_signal(self._filter_rts2_signal)
            else:
                self.send_signal(self._filter_rts_signal)
        else:
            log.error(f"{self.node.name}: Too few / many qubits => {len(outPort)}")
            self.send_signal(self._filter_rts_signal)
            return False

        evtFH = self.await_signal(sender=self.FilterH_protocol, signal_label=self._filter_cts_signal)
        evtFV = self.await_signal(sender=self.FilterV_protocol, signal_label=self._filter_cts_signal)
        evtRST = self.await_signal(sender=self.SignalProtocol, signal_label=self._complete_signal)
        evt_expr = yield evtRST | (evtFH & evtFV)
        if evt_expr.first_term.value:
            log.warning(f"{self.node.name}: Resetting ...")
            return False
        log.debug(f"{self.node.name}: Got a CTS signal - sending to Filter")

        if len(qubit) == 1:
            self.node.ports[outPort[0]].tx_output(qubit[0])
        elif len(outPort) == 2:
            if outPort[0] == outPort[1]:
                self.node.ports[outPort[0]].tx_output([qubit[0],qubit[1]])
            else:
                self.node.ports[outPort[0]].tx_output(qubit[0])
                self.node.ports[outPort[1]].tx_output(qubit[1])

        log.debug(f"{self.node.name}: Sent Qubit to DWDM Filter")

        evtRFH = self.await_signal(sender=self.FilterH_protocol, signal_label=self._qubit_recv_signal)
        evtRFV = self.await_signal(sender=self.FilterV_protocol, signal_label=self._qubit_recv_signal)
        evt_expr = yield evtRST | (evtRFH | evtRFV)
        if evt_expr.first_term.value:
            log.debug(f"{self.node.name}: Resetting ...")
        else:
            log.debug(f"{self.node.name}: Qubit received by {evt_expr.triggered_events[-1].source.name}")

        return True

class BeamSplitterProtocol(NodeProtocol):
    """
    A NetSquid component that simulates entanglement generation via a beam splitter,
    including the effects of spectral and temporal mismatch.

    The component models the physical indistinguishability of photons by calculating
    a total HOM visibility, which is then used to determine the level of noise
    in the entanglement operation.

    Insertion Loss is also factored in.

    Parameters
    ----------
    fwhm_ghz : float, optional
        The spectral Full-Width at Half-Maximum of the source photons in GHz.
        Default is 30.0 GHz.
    fwhm_duration_ps : float, optional
        The temporal FWHM duration of the source photons in picoseconds.
        If None, it is estimated from the fwhm_ghz as a transform-limited pulse.
    """

    def __init__(self, node, name=None, fwhm_ghz=30.0, fwhm_duration_ps=None, insertion_loss_dB=0.0):
        super().__init__(node=node, name=name)
        self._qubit_recv_signal = "qubit_recv"
        self._qubit_sent_signal = "qubit_sent"
        self._qubit_rts_signal = "qubit_rts"
        self._qubit_cts_signal = "qubit_cts"
        self._qubit_ent_signal = "qubit_ent"
        self._qubit_nent_signal = "qubit_nent"
        self._pbs_rts_signal = "pbs_rts"
        self._pbs_cts_signal = "pbs_cts"
        self._pbs_rts2_signal = "pbs_rts2"
        self._pbs_cts2_signal = "pbs_cts2"
        self._qubit_lost_signal = "qubit_lost"
        self._complete_signal = "complete"
        self._reset_signal = "reset"
        self.add_signal(self._reset_signal)
        self.add_signal(self._complete_signal)
        self.add_signal(self._qubit_lost_signal)
        self.add_signal(self._pbs_rts_signal)
        self.add_signal(self._pbs_cts_signal)
        self.add_signal(self._pbs_rts2_signal)
        self.add_signal(self._pbs_cts2_signal)
        self.add_signal(self._qubit_ent_signal)
        self.add_signal(self._qubit_nent_signal)
        self.add_signal(self._qubit_recv_signal)
        self.add_signal(self._qubit_sent_signal)
        self.add_signal(self._qubit_rts_signal)
        self.add_signal(self._qubit_cts_signal)

        # Required for 2 phase initialisation - due to circular dependencies.
        self.SPDCa_protocol = None
        self.SPDCb_protocol = None
        self.PBSa_protocol = None
        self.PBSb_protocol = None


    def connect(self, spdca_protocol, spdcb_protocol, PBSa_protocol, PBSb_protocol):
        """Wire up the connections to other protocols."""
        #print(f"{self.name}: Connecting to other protocols.")
        self.SPDCa_protocol = spdca_protocol
        self.SPDCb_protocol = spdcb_protocol
        self.PBSa_protocol = PBSa_protocol
        self.PBSb_protocol = PBSb_protocol

    def disconnect(self):
        self.SPDCa_protocol = None
        self.SPDCb_protocol = None
        self.PBSa_protocol = None
        self.PBSb_protocol = None

    def run(self):
        loaded = False
        entangled = False
        # Receive qubits from SPDC sources, apply insertion losses, and load into memory
        loaded = yield from self._receive_and_load_qubits_to_memory()

        # Entangle loaded qubits, apply deploraisation noise, and determine HOM Visibility
        if loaded:
            entangled, visibility = yield from self._entangle_qubits()

        if entangled:
            yield from self._transmit_qubits(visibility)

    def _receive_and_load_qubits_to_memory(self):
        log.debug(f"###   {self.node.name} restarting   ###")
        Aqport = self.node.ports["spdcaIN"]
        Bqport = self.node.ports["spdcbIN"]
        log.debug(f"{self.node.name}: Waiting for a signal")
        evt_rtsA = self.await_signal(sender=self.SPDCa_protocol,signal_label=self._qubit_rts_signal)
        evt_rtsB = self.await_signal(sender=self.SPDCb_protocol,signal_label=self._qubit_rts_signal)
        yield evt_rtsA & evt_rtsB
        log.debug(f"{self.node.name}: Got a RTS signal")
        self.send_signal(self._qubit_cts_signal)
        #
        # Should refactor this wait to be parallel not sequential
        #
        yield self.await_port_input(Aqport)
        log.debug(f"{self.node.name}: Received something from SPDC a")
        qa = Aqport.rx_input().items
        yield self.await_port_input(Bqport)
        log.debug(f"{self.node.name}: Received something from SPDC b")
        qb = Bqport.rx_input().items
        self.send_signal(self._qubit_recv_signal)
        evt_spdca = self.await_signal(sender=self.SPDCa_protocol, signal_label=self._qubit_sent_signal)
        evt_spdcb = self.await_signal(sender=self.SPDCb_protocol, signal_label=self._qubit_sent_signal)
        yield evt_spdca & evt_spdcb
        log.debug(f"{self.node.name}: Got a SENT signal from SPDCa and b")
        #
        # Applying effiency factor here : INSERTION LOSS
        #
        loss_prob = 1 - 10**(-config.BEAMSPLITTER_INSERTION_LOSS_DB / 10)
        if np.random.rand() < loss_prob or np.random.rand() < loss_prob: # 2 qubits
            qapi.discard(qa[0])
            qapi.discard(qb[0])
            log.warning(f"{self.node.name}: Insertion Loss")
            result_payload = { 'status': "ERROR", 'error_type': "INSERTION Loss", 'location': self.node.name}
            self.send_signal(self._reset_signal, result=result_payload)
            return False
        # Randomly assign memory location
        # For random output assignment
        set_mem_zero = random.choice([qa[0], qb[0]])
        if set_mem_zero.name == qa[0].name:
            self.node.qmemory.put([qa[0], qb[0]])
        else:
            self.node.qmemory.put([qb[0], qa[0]])
        log.debug(f"Put Qubits {qa[0].name} and {qb[0].name} into memory")

        return True

    def _entangle_qubits(self):
        q1, q2 = self.node.qmemory.peek(positions=[0, 1], skip_noise=True)
        if q1:
            log.debug(f"{self.node.name}: SPDCa Frequency is {q1.wavelength} and SPDCb Frequency is {q2.wavelength}")
            log.debug(f"{self.node.name}: SPDCa Polarisation is {q1.polarisation} and SPDCb Polarisation is {q2.polarisation}")
        else:
            log.error(f"{self.node.name}: No qubit in memory location 0")

        cnot_qubits = CNOTQubits()
        yield self.node.qmemory.execute_program(cnot_qubits)
        self.send_signal(self._qubit_ent_signal)
        log.debug(f"{self.node.name}: Entangled Qubits at the BeamSplitter")

        # Applying any post entanglement depolarisation - may remove entanglement
        # Calculating HOM Visibility
        delta_f_ghz = abs(q1.wavelength - q2.wavelength)/1e9 # Convert from Hz to GHz
        delta_t_ps = 0.0 # could be a random number to introduce jitter - need to load from config file

        if not config.HOM_VISIBILITY_AT_MAX:
            visibility = calculate_visibility(delta_f_ghz, delta_t_ps)
        else:
            visibility = 1
        depolarizing_prob = 1 - visibility
        if visibility < 0.5:
            log.warning(f"{self.node.name}: HOM Visibility too low : {visibility}")
            result_payload = { 'status': "ERROR", 'error_type': "LOW Visibility", 'location': self.node.name, 'visibility': visibility }
            self.send_signal(self._reset_signal, result=result_payload)
        log.debug(f"{self.node.name}: Depolarisation Probability is {depolarizing_prob}")
        log.debug(f"{self.node.name}: Entangled Before Depolarisation?: {q1.qstate == q2.qstate}")

        # Using e-9 to mitigate round-off errors
        if depolarizing_prob > 1e-9:
            noisy1, noisy2 = self.node.qmemory.pop([0, 1])
            qapi.depolarize(noisy1, prob=depolarizing_prob)
            qapi.depolarize(noisy2, prob=depolarizing_prob)
            self.node.qmemory.put([noisy1, noisy2])
        q1, q2 = self.node.qmemory.peek(positions=[0, 1], skip_noise=True)
        #log.debug(f"{self.node.name}: Fidelity of generated entanglement: {ns.qubits.fidelity([q1, q2], ks.b01)}")
        log.debug("Beamsplitter: Fidelity of generated entanglement {}".format(round(ns.qubits.fidelity([q1, q2], ks.b11))))
        return True, visibility

    def _transmit_qubits(self, visibility):
        # Indistinguishable photons will always exit on the same port (bunching)
        # Distinguishable photons have a 50:50 chance of bunching
        q1, q2 = self.node.qmemory.peek(positions=[0, 1], skip_noise=True)
        will_bunch = False
        if (q1.polarisation == q2.polarisation) & (visibility > config.BEAMSPLITTER_HOM_THRESHOLD): #indistinguishable
            will_bunch = True
        else: #distinguishable
            will_bunch = random.choice([False, True])
        log.debug(f"{self.node.name}: Photons will exit the same port: {will_bunch}")

        # Apply Phase Flip on reflected port - "PBSaOUT"
        flip_qubits = []
        if will_bunch:
            port = random.choice(["PBSaOUT", "PBSbOUT"])
            log.debug(f"Bunching qubits on port {port}")
            if port == "PBSaOUT":
                flip_qubits = [0,1]
        else:
            flip_qubits = [0]

        if flip_qubits != []:
            phaseFlip = PhaseFlipQubits()
            yield self.node.qmemory.execute_program(phaseFlip, flipQubits=flip_qubits)

        try:
            qubit1 = self.node.qmemory.pop(0)
        except:
            log.warning(f"{self.node.name}: Emmission Loss : Qubit failed to pop")
            result_payload = { 'status': "ERROR", 'error_type': "EMMISSION LOSS", 'location': self.node.name}
            self.send_signal(self._reset_signal, result=result_payload)
            Error = True
            return False
        log.info(f"{self.node.name}: Result: {qubit1[0].polarisation}")
        try:
            qubit2 = self.node.qmemory.pop(1)
        except:
            log.warning(f"{self.node.name}: Emmission Loss : Qubit failed to pop")
            result_payload = { 'status': "ERROR", 'error_type': "EMMISSION LOSS", 'location': self.node.name}
            self.send_signal(self._reset_signal, result=result_payload)
            Error = True
            return False
        log.info(f"{self.node.name}: Result: {qubit2[0].polarisation}")

        if any(qubit.qstate is None for qubit in [qubit1[0], qubit2[0]]):
            log.error(f"{self.node.name}: Missing a qubit")

        if will_bunch:
            log.debug(f"{self.node.name}: Sending CTS2 signal from {port}")
            self.send_signal(self._pbs_rts2_signal)
            if port == "PBSaOUT":
                yield self.await_signal(sender=self.PBSa_protocol, signal_label=self._pbs_cts2_signal)
                log.debug(f"{self.node.name}: Transmitting 2 qubits to {self.PBSa_protocol.node.name}")
                self.node.ports["PBSaOUT"].tx_output([qubit1[0],qubit2[0]])
                yield self.await_signal(sender=self.PBSa_protocol, signal_label=self._qubit_recv_signal)
                log.info(f"{self.node.name}: Looks like PBS {self.PBSa_protocol.node.name} received two emitted qubits")
            elif port == "PBSbOUT":
                yield self.await_signal(sender=self.PBSb_protocol, signal_label=self._pbs_cts2_signal)
                log.debug(f"{self.node.name}: Transmitting 2 qubits to {self.PBSb_protocol.node.name}")
                self.node.ports["PBSbOUT"].tx_output([qubit1[0],qubit2[0]])
                yield self.await_signal(sender=self.PBSb_protocol, signal_label=self._qubit_recv_signal)
                log.info(f"{self.node.name}: Looks like PBS {self.PBSb_protocol.node.name} received two emitted qubits")
        else:
            log.debug(f"{self.node.name}: Sending CTS signal")
            self.send_signal(self._pbs_rts_signal)
            evt_PBSa = self.await_signal(sender=self.PBSa_protocol, signal_label=self._pbs_cts_signal)
            evt_PBSb = self.await_signal(sender=self.PBSb_protocol, signal_label=self._pbs_cts_signal)
            yield evt_PBSa & evt_PBSb
            self.node.ports["PBSaOUT"].tx_output(qubit1[0])
            self.node.ports["PBSbOUT"].tx_output(qubit2[0])
            evt_PBSarecv = self.await_signal(sender=self.PBSa_protocol, signal_label=self._qubit_recv_signal)
            evt_PBSbrecv = self.await_signal(sender=self.PBSb_protocol, signal_label=self._qubit_recv_signal)
            yield evt_PBSarecv & evt_PBSbrecv
            log.info(f"{self.node.name}: Looks like both PBSs received an emitted qubit")


class FilterProtocol(NodeProtocol):

    def __init__(self, node, name=None):
        super().__init__(node=node, name=name)
        self._qubit_recv_signal = "qubit_recv"
        #self._qubit_sent_signal = "qubit_sent"
        self._qubit_rts_signal = "qubit_rts"
        self._qubit_cts_signal = "qubit_cts"
        self._qubit_rts2_signal = "qubit_rts2"
        self._qubit_cts2_signal = "qubit_cts2"
        self._qubit_ent_signal = "qubit_ent"
        self._filter_rts_signal = "filter_rts"
        self._filter_cts_signal = "filter_cts"
        self._qubit_nent_signal = "qubit_nent"
        self._qubit_lost_signal = "qubit_lost"
        self._pbs_rts_signal = "pbs_rts"
        self._pbs_cts_signal = "pbs_cts"
        self._qubit_lost_signal = "qubit_lost"
        self._complete_signal = "complete"
        self._filter_rts2_signal = "filter_rts2"
        self._reset_signal = "reset"
        self.add_signal(self._reset_signal)
        self.add_signal(self._filter_rts2_signal)
        self.add_signal(self._complete_signal)
        self.add_signal(self._qubit_lost_signal)
        self.add_signal(self._pbs_rts_signal)
        self.add_signal(self._pbs_cts_signal)
        self.add_signal(self._qubit_ent_signal)
        self.add_signal(self._qubit_nent_signal)
        self.add_signal(self._qubit_recv_signal)
        #self.add_signal(self._qubit_sent_signal)
        self.add_signal(self._qubit_rts_signal)
        self.add_signal(self._qubit_cts_signal)
        self.add_signal(self._qubit_rts2_signal)
        self.add_signal(self._qubit_cts2_signal)
        self.add_signal(self._filter_rts_signal)
        self.add_signal(self._filter_cts_signal)
        self.PBS_protocol = None
        self.HS_protocol = None
        self.SignalProtocol = None

    def connect(self, PBS_protocol, HS_protocol, S_protocol):
        """Wire up the connections to other protocols."""
        #print(f"{self.name}: Connecting to other protocols.")
        self.PBS_protocol = PBS_protocol
        self.HS_protocol = HS_protocol
        self.SignalProtocol = S_protocol

    def disconnect(self):
        self.PBS_protocol = None
        self.HS_protocol = None
        self.SignalProtocol = None

    def run(self):
        detectable = False
        detectable, numqubits = yield from self._dwdm_receive_photons()

        if detectable:
            yield from self._route_to_detector_and_signal_heralding_station(numqubits)

        return True

    def _dwdm_receive_photons(self):
        log.debug(f"###   {self.node.name} restarting   ###")
        num_qubits = 1
        evtRST = self.await_signal(sender=self.SignalProtocol, signal_label=self._complete_signal)
        qport = self.node.ports["qin"]
        evt1q = self.await_signal(sender=self.PBS_protocol, signal_label=self._filter_rts_signal)
        evt2q = self.await_signal(sender=self.PBS_protocol, signal_label=self._filter_rts2_signal)
        evtnumq = yield evt1q | evt2q
        if evtnumq.second_term.value:
            num_qubits = 2
        self.send_signal(self._filter_cts_signal)
        log.debug(f"{self.node.name} received an RTS and sent a CTS")
        evt_expr = yield self.await_port_input(qport) | self.await_signal(sender=self.SignalProtocol, signal_label=self._complete_signal)
        if evt_expr.second_term.value:
            return False, 1
        qubitPBS = qport.rx_input().items
        log.debug(f"{self.node.name} received a qubit from the Polarisation BeamSplitter")
        if len(qubitPBS) != num_qubits:
            log.error(f"{self.node.name}: Expected {num_qubits} only received {len(qubitPBS)}")
            return False, 1

        # --- DWDM FIlter Insertion Loss ---
        loss_prob = 1 - 10**(-config.DWDM_FILTER_INSERTION_LOSS_DB / 10)
        if len(qubitPBS) == 1:
            if np.random.rand() < loss_prob:
                qapi.discard(qubitPBS[0])
                result_payload = { 'status': "ERROR", 'error_type': "INSERTION Loss", 'location': self.node.name}
                self.send_signal(self._reset_signal, result=result_payload)
                return False, 1
            self.node.qmemory.put(qubitPBS[0])
        else:
            if np.random.rand() < loss_prob or np.random.rand() < loss_prob: # 2 qubits
                qapi.discard(qubitPBS[0])
                qapi.discard(qubitPBS[1])
                result_payload = { 'status': "ERROR", 'error_type': "INSERTION Loss", 'location': self.node.name}
                self.send_signal(self._reset_signal, result=result_payload)
                return False, 1
            self.node.qmemory.put([qubitPBS[0],qubitPBS[1]])
        self.send_signal(self._qubit_recv_signal)
        log.debug(f"{self.node.name}: {len(qubitPBS)} qubits received")

        return True, len(qubitPBS)


    def _route_to_detector_and_signal_heralding_station(self, numqubits):
        passband_fwhm_ghz = config.GRID_GRANULARITY_GHZ * config.FILTER_PASSBAND_FRACTION


        evtRST = self.await_signal(sender=self.SignalProtocol, signal_label=self._complete_signal)
        pos=0
        for _ in range(numqubits):
            qubit = self.node.qmemory.peek(positions=pos, skip_noise=True)
            log.debug(f"{self.node.name}: Peeked qubit is {qubit[0].wavelength}")
            ch = find_nearest_channel(qubit[0].wavelength)
            log.info(f"{self.node.name}: Qubit recived at filter {self.node.name} on channel {ch}")
            if numqubits == 1:
                self.send_signal(self._qubit_rts_signal, result=qubit[0].polarisation)
                evtxpr = yield evtRST | self.await_signal(sender=self.HS_protocol, signal_label=self._qubit_cts_signal)
                if evtxpr.first_term.value:
                    log.warning(f"{self.node.name}: Resetting ...")
                    return False
            else:
                self.send_signal(self._qubit_rts2_signal, result=qubit[0].polarisation)
                evtxpr = yield evtRST | self.await_signal(sender=self.HS_protocol, signal_label=self._qubit_cts2_signal)
                if evtxpr.first_term.value:
                    log.warning(f"{self.node.name}: Resetting ...")
                    return False
            log.debug(f"{self.node.name}: Got a CTS signal - sending to Heralding Station")
            try:
                res = self.node.qmemory.pop(pos)
            except:
                log.warning(f"{self.node.name}: EMMISSION Loss : Qubit failed to pop")
                result_payload = { 'status': "ERROR", 'error_type': "EMMISSION LOSS", 'location': self.node.name}
                self.send_signal(self._reset_signal, result=result_payload)
                return False
            if any(qubit.qstate is None for qubit in [res[0]]):
                log.error(f"{self.node.name}: Missing a qubit")

            # This part assumes we know the center freq of this specific filter.
            # A more robust way would be to get it from the node name or a passed-in param.
            # For now, we find the closest channel center.
            center_channel_idx = find_channel_from_freq(qubit[0].wavelength * 1e-12)
            channel_center_ghz = find_freq_from_channel(center_channel_idx)*1000
            #next(freq for freq, idx in config.DWDM_CHANNELS.items() if idx == center_channel_idx) * 1000
            photon_freq_ghz = qubit[0].wavelength * 1e-9 # convert Hz to GHz
            delta_f_ghz = abs(photon_freq_ghz - channel_center_ghz)

            log.debug(f"Channel ID: {center_channel_idx}, channel centre freq {channel_center_ghz}, photon_freq {photon_freq_ghz}, delta {delta_f_ghz}")

            transmission_prob = 1.0
            if config.FILTER_MODEL == 'BRICK_WALL':
                if delta_f_ghz > passband_fwhm_ghz / 2:
                    transmission_prob = 0.0
            elif config.FILTER_MODEL == 'GAUSSIAN':
                sigma_f = passband_fwhm_ghz / (2 * np.sqrt(2 * np.log(2)))
                transmission_prob = np.exp(-(delta_f_ghz**2) / (2 * sigma_f**2))

            if np.random.rand() > transmission_prob:
                log.info(f"FILTER LOSS at {self.node.name}: Photon freq blocked.")
                qapi.discard(qubit[0])
                result_payload = { 'status': "ERROR", 'error_type': "Photon Blocked", 'location': self.node.name}
                self.send_signal(self._reset_signal, result=result_payload)
                return

            # --- Detector Efficiency ---
            if np.random.rand() < 1 - config.DETECTOR_EFFICIENCY:
                #log.info(f"CLICK! at detector {self.node.name}")
                #self.heralding_station.send_signal(DETECTION_SIGNAL, result=self.detector_name)
                qapi.discard(qubit[0])
                result_payload = { 'status': "ERROR", 'error_type': "Detector Loss", 'location': self.node.name}
                self.send_signal(self._reset_signal, result=result_payload)
                return

            self.node.ports["qout"].tx_output(res[0])
            log.info(f"{self.node.name}: Result: {res[0].polarisation}")
            pos += 1

        return True


class HeraldingProtocol(NodeProtocol):

    def __init__(self, node, name=None):
        super().__init__(node=node, name=name)
        self._qubit_recv_signal = "qubit_recv"
        #self._qubit_sent_signal = "qubit_sent"
        self._qubit_rts_signal = "qubit_rts"
        self._qubit_cts_signal = "qubit_cts"
        self._qubit_rts2_signal = "qubit_rts2"
        self._qubit_cts2_signal = "qubit_cts2"
        self._filter_rts_signal = "filter_rts"
        self._filter_cts_signal = "filter_cts"
        self._qubit_ent_signal = "qubit_ent"
        self._qubit_nent_signal = "qubit_nent"
        self._qubit_lost_signal = "qubit_lost"
        self._pbs_rts_signal = "pbs_rts"
        self._pbs_cts_signal = "pbs_cts"
        self._qubit_lost_signal = "qubit_lost"
        self._postp_rts_signal = "postp_rts"
        self._postp_cts_signal = "postp_cts"
        self._measure_signal = "measure"
        self._processed_signal = "processed"
        self._complete_signal = "complete"
        self._bell_state_signal ="bellState"
        self._reset_signal = "reset"
        self.add_signal(self._reset_signal)
        self.add_signal(self._bell_state_signal)
        self.add_signal(self._complete_signal)
        self.add_signal(self._qubit_lost_signal)
        self.add_signal(self._pbs_rts_signal)
        self.add_signal(self._pbs_cts_signal)
        self.add_signal(self._qubit_ent_signal)
        self.add_signal(self._qubit_nent_signal)
        self.add_signal(self._qubit_recv_signal)
        #self.add_signal(self._qubit_sent_signal)
        self.add_signal(self._qubit_rts_signal)
        self.add_signal(self._qubit_rts2_signal)
        self.add_signal(self._qubit_cts_signal)
        self.add_signal(self._qubit_cts2_signal)
        self.add_signal(self._filter_rts_signal)
        self.add_signal(self._filter_cts_signal)
        self.add_signal(self._postp_rts_signal)
        self.add_signal(self._postp_cts_signal)
        self.add_signal(self._measure_signal)
        self.add_signal(self._processed_signal)
        self.FilterAH_protocol = None
        self.FilterAV_protocol = None
        self.FilterBH_protocol = None
        self.FilterBV_protocol = None
        self.PostPA_protocol = None
        self.PostPB_protocol = None

    def connect(self, filterAH_protocol, filterAV_protocol, filterBH_protocol, filterBV_protocol, postA_protocol, postB_protocol):
        """Wire up the connections to other protocols."""
        #print(f"{self.name}: Connecting to other protocols.")
        self.FilterAH_protocol = filterAH_protocol
        self.FilterAV_protocol = filterAV_protocol
        self.FilterBH_protocol = filterBH_protocol
        self.FilterBV_protocol = filterBV_protocol
        self.PostPA_protocol = postA_protocol
        self.PostPB_protocol = postB_protocol

    def disconnect(self):
        self.FilterAH_protocol = None
        self.FilterAV_protocol = None
        self.FilterBH_protocol = None
        self.FilterBV_protocol = None
        self.PostPA_protocol = None
        self.PostPB_protocol = None

    def run(self):
        Error = False
        log.debug(f"###   {self.node.name} restarting   ###")
        entFAH = self.await_signal(sender=self.FilterAH_protocol, signal_label=self._qubit_rts_signal)
        entFAV = self.await_signal(sender=self.FilterAV_protocol, signal_label=self._qubit_rts_signal)
        entFBH = self.await_signal(sender=self.FilterBH_protocol, signal_label=self._qubit_rts_signal)
        entFBV = self.await_signal(sender=self.FilterBV_protocol, signal_label=self._qubit_rts_signal)
        entFAH2 = self.await_signal(sender=self.FilterAH_protocol, signal_label=self._qubit_rts2_signal)
        entFAV2 = self.await_signal(sender=self.FilterAV_protocol, signal_label=self._qubit_rts2_signal)
        entFBH2 = self.await_signal(sender=self.FilterBH_protocol, signal_label=self._qubit_rts2_signal)
        entFBV2 = self.await_signal(sender=self.FilterBV_protocol, signal_label=self._qubit_rts2_signal)
        evt_expr = yield (entFAH & entFBH) | (entFAV & entFBV) | (entFAH & entFBV) | (entFAV & entFBH) | (entFAH & entFAV) | (entFBH & entFBV) | entFAH2 | entFBH2 | entFAV2 | entFBV2
        Filter1 = evt_expr.triggered_events[0].source.node.name[-2:]
        Filter2 = evt_expr.triggered_events[-1].source.node.name[-2:]
        log.debug(f"{self.node.name}: Detections received on {Filter1} and {Filter2}")
        bellState = determineBellState(Filter1, Filter2)
        log.info(bellState)
        if Filter1 == Filter2: #PHI States
            counter = 0
            if config.DETECTOR_TYPE != 'PNR':
                log.warning(f"{self.node.name}: Incorrect Detector Type for PHI states")
                result_payload = { 'status': "ERROR", 'error_type': "Undetectable state", 'location': self.node.name}
                self.send_signal(self._reset_signal, result=result_payload)
                return False
            while counter < 2:
                self.send_signal(self._qubit_cts2_signal)
                qport=self.node.ports[f"qin{Filter1}"]
                yield self.await_port_input(qport)
                qubits = qport.rx_input().items
                log.info(f"{self.node.name}: Received {qubits}")
                try:
                    self.node.qmemory.put(qubits[0], positions=counter)
                    counter += 1
                except:
                    log.warning(f"{self.node.name}: Insertion Loss : Qubit failed to put")
                    result_payload = { 'status': "ERROR", 'error_type': "INSERTION LOSS", 'location': self.node.name}
                    self.send_signal(self._reset_signal, result=result_payload)
                    return False
        else:
            self.send_signal(self._qubit_cts_signal)
            qport1=self.node.ports[f"qin{Filter1}"]
            qport2=self.node.ports[f"qin{Filter2}"]
            yield self.await_port_input(qport1) & self.await_port_input(qport2)
            qubit1 = qport1.rx_input().items
            qubit2 = qport2.rx_input().items
            log.info(f"{self.node.name}: Received {qubit1} and {qubit2}")
            try:
                self.node.qmemory.put([qubit1[0], qubit2[0]])
            except:
                log.warning(f"{self.node.name}: Insertion Loss : Qubit failed to put")
                result_payload = { 'status': "ERROR", 'error_type': "INSERTION LOSS", 'location': self.node.name}
                self.send_signal(self._reset_signal, result=result_payload)
                return False

        #self.send_signal(self._qubit_recv_signal)
        bsm_program = BellMeasurementProgram()
        yield self.node.qmemory.execute_program(bsm_program)
        # Retrieve the measurement outcomes
        m1, = bsm_program.output["M1"]
        m2, = bsm_program.output["M2"]

        # Output the results
        log.info(f"Q2 measurement: {m1}")
        log.info(f"Q4 measurement: {m2}")

        # PostProcess Signal Qubits
        yield from self._postProcess(m1, m2)
        self.send_signal(self._bell_state_signal, result=bellState)

        return True

    def _postProcess(self, m1, m2):
        outcome1 = m1
        outcome2 = m2
        self.send_signal(self._postp_rts_signal)
        evtPPA = self.await_signal(sender=self.PostPA_protocol, signal_label=self._postp_cts_signal)
        evtPPB = self.await_signal(sender=self.PostPB_protocol, signal_label=self._postp_cts_signal)
        yield evtPPA & evtPPB
        messageA="None"
        messageB="None"
        if outcome1 == 0 and outcome2 == 0:
            # Apply a Z gate to qubit Signal1 and an X gate to Signal2
            messageA="Z"
            messageB="X"
            log.debug(f"Z gate for PPA and X Gate for PPB")
        elif outcome1 == 0 and outcome2 == 1:
            # Apply an X gate to Signal2
            messageB="X"
            log.debug(f"None action for PPA and X Gate for PPB")
        elif outcome1 == 1 and outcome2 == 0:
            # Apply a Z gate to qubit Signal1
            messageA="Z"
            log.debug(f"Z gate for PPA and None action PPB")
        elif outcome1 == 1 and outcome2 == 1:
            # No operations needed, qubits are already in the psi- state
            # pass
            log.debug(f"None action for PPA and PPB")
        self.node.ports["coutPPA"].tx_output(messageA)
        self.node.ports["coutPPB"].tx_output(messageB)
        evtProcA = self.await_signal(sender=self.PostPA_protocol, signal_label=self._processed_signal)
        evtProcB = self.await_signal(sender=self.PostPB_protocol, signal_label=self._processed_signal)
        yield evtProcA & evtProcB
        log.debug(f"{self.node.name}: Requesting Post processor to check their qubits fidelity")
        self.send_signal(self._measure_signal)


class PostProcessorProtocol(NodeProtocol):
    """
    Delay line for the Signal qubits and updates their state to represent desired state.
    """

    def __init__(self, node, name=None, fwhm_ghz=30.0, fwhm_duration_ps=None, insertion_loss_dB=0.0):
        super().__init__(node=node, name=name)
        self._qubit_recv_signal = "qubit_recv"
        self._qubit_sent_signal = "qubit_sent"
        self._qubit_rts_signal = "qubit_rts"
        self._qubit_cts_signal = "qubit_cts"
        self._qubit_ent_signal = "qubit_ent"
        self._qubit_nent_signal = "qubit_nent"
        self._pbs_rts_signal = "pbs_rts"
        self._pbs_cts_signal = "pbs_cts"
        self._qubit_lost_signal = "qubit_lost"
        self._postp_rts_signal = "postp_rts"
        self._postp_cts_signal = "postp_cts"
        self._measure_signal = "measure"
        self._processed_signal = "processed"
        self._complete_signal = "complete"
        self.add_signal(self._complete_signal)
        self.add_signal(self._qubit_lost_signal)
        self.add_signal(self._pbs_rts_signal)
        self.add_signal(self._pbs_cts_signal)
        self.add_signal(self._qubit_ent_signal)
        self.add_signal(self._qubit_nent_signal)
        self.add_signal(self._qubit_recv_signal)
        self.add_signal(self._qubit_sent_signal)
        self.add_signal(self._qubit_rts_signal)
        self.add_signal(self._qubit_cts_signal)
        self.add_signal(self._postp_rts_signal)
        self.add_signal(self._postp_cts_signal)
        self.add_signal(self._measure_signal)
        self.add_signal(self._processed_signal)
        # Required for 2 phase initialisation - due to circular dependencies.
        self.SPDC_protocol = None
        self.HS_protocol = None
        self.SignalProtocol = None

    def connect(self, spdc_protocol, HS_protocol, signal_protocol):
        """Wire up the connections to other protocols."""
        #print(f"{self.name}: Connecting to other protocols.")
        self.SPDC_protocol = spdc_protocol
        self.HS_protocol = HS_protocol
        self.SignalProtocol = signal_protocol

    def disconnect(self):
        self.SPDC_protocol = None
        self.HS_protocol = None
        self.SignalProtocol = None

    def run(self):
        log.debug(f"###   {self.node.name} restarting   ###")
        # Load Signal qubits into memory
        qport = self.node.ports["qin"]
        cport = self.node.ports["cHS"]
        log.debug(f"{self.node.name}: Waiting for a signal")
        yield self.await_signal(sender=self.SPDC_protocol,signal_label=self._postp_rts_signal)
        log.debug(f"{self.node.name}: Got a RTS signal {self.SPDC_protocol.node.name}")
        self.send_signal(self._postp_cts_signal)

        yield self.await_port_input(qport)
        log.debug(f"{self.node.name}: Received something from {self.SPDC_protocol.node.name}")
        qubit = qport.rx_input().items
        self.node.qmemory.put(qubit[0])
        sig = self.node.qmemory.peek(positions=[0], skip_noise=True)
        #print(f"{self.node.name}: Signal Qubit State is now {sig[0].qstate.qrepr} ")
        log.debug(f"{self.node.name}: Loaded qubit {sig[0]} into memory")
        self.send_signal(self._qubit_recv_signal)

        evt_expr = yield self.await_signal(sender=self.HS_protocol,signal_label=self._postp_rts_signal) | self.await_signal(sender=self.SignalProtocol, signal_label=self._complete_signal)
        if evt_expr.second_term.value:
            # Clean up and continue
            log.warning(f"{self.node.name}: Cleaning up")
            #self.node.qmemory.pop(0)
            self.node.qmemory.reset()
            return # quit running
        log.debug(f"{self.node.name}: Got a RTS signal from {self.HS_protocol.node.name}")
        self.send_signal(self._postp_cts_signal)

        yield self.await_port_input(cport)
        message = cport.rx_input().items
        action = message[0]
        log.debug(f"{self.node.name}: Received Message {action} from {self.HS_protocol.node.name}")
        post_process_program = PostProcessProgram()
        yield self.node.qmemory.execute_program(post_process_program, gate=action)
        self.send_signal(self._processed_signal)
        yield self.await_signal(sender=self.HS_protocol,signal_label=self._measure_signal)
        log.debug(f"{self.node.name}: Sending Qubit for Measuring ...")
        qubit = self.node.qmemory.pop(0)
        self.node.ports["qout"].tx_output(qubit[0])

class MeasureStationProtocol(NodeProtocol):
    def __init__(self, node, name=None):
        super().__init__(node=node, name=name)
        self._complete_signal = "complete"
        self._reset_signal = "reset"
        self.add_signal(self._reset_signal)
        self.add_signal(self._complete_signal)
        self.SignalProtocol = None

    def connect(self, S_protocol):
        """Wire up the connections to other protocols."""
        #print(f"{self.name}: Connecting to other protocols.")
        self.SignalProtocol = S_protocol

    def disconnect(self):
        self.SignalProtocol = None

    def run(self):
        # reset variables to None
        qportA = qportB = qubitA = qubitB = q1 = q2 = fidelity = status = result_payload = None
        log.debug(f"###   {self.node.name} restarting   ###")
        qportA = self.node.ports["qinA"]
        qportB = self.node.ports["qinB"]
        evt_expr = yield self.await_port_input(qportA) & self.await_port_input(qportB) # | self.await_signal(sender=self.SignalProtocol, signal_label=self._complete_signal)
        log.debug(f"{self.node.name}: Measuring ...")
        qubitA = qportA.rx_input().items
        qubitB = qportB.rx_input().items
        q1 = qubitA[0]
        q2 = qubitB[0]
        log.debug("Fidelity (B00 -> PHI+) of final entanglement: {}".format(round(ns.qubits.fidelity([q1, q2], ks.b00, squared=True))))
        log.debug("Fidelity (B01 -> PSI+) of final entanglement: {}".format(round(ns.qubits.fidelity([q1, q2], ks.b01, squared=True))))
        log.debug("Fidelity (B10 -> PHI-) of final entanglement: {}".format(round(ns.qubits.fidelity([q1, q2], ks.b10, squared=True))))
        log.debug("Fidelity (B11 -> PSI-) of final entanglement: {}".format(round(ns.qubits.fidelity([q1, q2], ks.b11, squared=True))))
        # --- Fidelity Calculation with Warning Suppression --- Doesn't work
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.linalg.LinAlgError)
            #fidelity = ns.qubits.fidelity([q1, q2], ks.b11, squared=True)
            fidelity = ns.qubits.fidelity([q1, q2], TARGET_PSIminus_DM, squared=True)
            #if round(ns.qubits.fidelity([q1, q2], ks.b11, squared=True)) > 0.5:
            if round(ns.qubits.fidelity([q1, q2], TARGET_PSIminus_DM, squared=True)) > 0.5:
                log.warning(f"Fidelity (B11 -> PSI-) of final entanglement: {fidelity}")
            log.debug(f"Fidelity (B11 -> PSI-) of final entanglement: {fidelity}")
            log.info(f"Qubit 1 is : {q1.wavelength*1e-9:.4f}")
            log.info(f"Qubit 2 is : {q2.wavelength*1e-9:.4f}")
            if abs(q1.wavelength*1e-9 - q2.wavelength*1e-9) < 20:
                log.info(f"Chance of entanglement : {q1.wavelength*1e-9:.4f} and {q2.wavelength*1e-9}")
            status = "SUCCESS"
            #if round(ns.qubits.fidelity([q1, q2], ks.b11, squared=True)) != 1:
            if round(ns.qubits.fidelity([q1, q2], TARGET_PSIminus_DM, squared=True)) != 1:
                log.debug(f"Qubit 1 is : {q1.qstate.qrepr}")
                log.debug(f"Qubit 2 is : {q2.qstate.qrepr}")
                status = "FAILURE"
        #result_payload = { "status": status, "fidelity": fidelity, "dwdm_channel": int(find_nearest_channel(q1.wavelength*1e-12)), "q1_state": q1.qstate.qrepr.reduced_dm(), "q2_state": q2.qstate.qrepr.reduced_dm(), "q1_frequency": q1.wavelength, "q2_frequency": q2.wavelength, "q1_polarisation": q1.polarisation, "q2_polarisation": q2.polarisation}
        result_payload = { "status": status, "fidelity": fidelity, "dwdm_channel": int(find_nearest_channel(q1.wavelength*1e-12)), "final_dm": serialize_qstate([q1, q2]), "q1_frequency": q1.wavelength, "q2_frequency": q2.wavelength, "q1_polarisation": q1.polarisation, "q2_polarisation": q2.polarisation, "q1_name": q1.name, "q2_name": q2.name}
        self.send_signal(self._complete_signal, result=result_payload)
