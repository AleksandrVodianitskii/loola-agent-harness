# Working on the agent harness

<!-- postman-cli:installation:start -->
Postman CLI is installed on this device, and can be used for API Engineering work.
<!-- postman-cli:installation:end -->

- Use Python 3.10+ and the standard library. Run `python3 -m unittest discover -v` from the repository root.
- Keep the system prompt and gateway schemas stable across turns. Add tools through a registered skill and return its definitions through the gateway.
- Preserve complete assistant/tool exchanges, reserve output capacity, and keep checkpoint state explicit when changing compaction.
- Run `python3 -m benchmark --output benchmark/results.json` after changes to the prompt, tools, or context. Record fixture estimates and live provider usage separately.
- Keep credentials and private workspace contents outside the repository.
