from functools import partial

from nemo_aligner.algorithms.math_grader_server import SimpleMathGrader, extract_and_check

ENDPOINT_BIND_ADDRESS = "0.0.0.0"


def main() -> None:
    server = SimpleMathGrader(
        grading_function=extract_and_check,
        port=5555,
        process_count=16,
    )
    server.run_server()


if __name__ == "__main__":
    main()
