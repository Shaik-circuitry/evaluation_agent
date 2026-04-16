"""Demo entry: hardcoded inputs, run evaluation, print report."""

from evaluator import get_store, run_evaluation
from evaluator.models import EvaluationInput, RetrievedChunk
from evaluator.report import print_result, print_summary

# --- INPUT VARIABLES --- change these
QUERY = "How do I reset the admin password on a Hypertec HX server when locked out of IPMI?"

HUMAN_RESPONSE = """To reset the admin password on your Hypertec HX IPMI when locked out, you need physical access.
1. Power off the server and remove power cables
2. Locate the CMOS reset jumper (typically J22 or CLR_CMOS in your HX model service manual)
3. Move jumper from pins 1–2 to pins 2–3 for 10 seconds, then back to 1–2
4. Reconnect power and boot — IPMI resets to factory default: admin / admin
5. Log in immediately and change the password under Configuration > Users
If no physical access, contact support@hypertec.com with your serial number."""

RAG_RESPONSE = """You can reset the IPMI admin password by accessing the server management console and using the password reset option in user management. If fully locked out, perform a factory reset. Refer to your documentation for steps."""

RETRIEVED_CHUNKS = [
    {
        "text": "Hypertec HX Quick Start — Section 4.2 Network Configuration. Default IPMI IP is 192.168.1.100. Default credentials on first boot: admin/admin. Navigate to Configuration > Network to assign static IP.",
        "relevance_score": 0.61,
    },
    {
        "text": "Hypertec Support FAQ — If IPMI is unreachable over network, check management port is connected and workstation is on same subnet. Some HX models require management NIC to be enabled in BIOS.",
        "relevance_score": 0.44,
    },
    {
        "text": "Hypertec HX Safety Guide — Power off and disconnect all power before hardware maintenance. Wear ESD wrist strap when handling internal components.",
        "relevance_score": 0.38,
    },
]


def main() -> None:
    chunks = [RetrievedChunk(**c) for c in RETRIEVED_CHUNKS]
    inp = EvaluationInput(
        query=QUERY,
        human_response=HUMAN_RESPONSE,
        rag_response=RAG_RESPONSE,
        retrieved_chunks=chunks,
        metadata={"demo": True},
    )
    result = run_evaluation(inp)
    print_result(result)
    print_summary(get_store())


if __name__ == "__main__":
    main()
