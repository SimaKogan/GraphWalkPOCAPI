from uuid import uuid4
import asyncio
from Core.routing_orchestrator import Orchestrator
from Config.settings import settings
from Visualization.display_utils import display_hebrew_markdown_pycharm


def main():
    exp_path = "Resources/SystemConfigs/exp_server_1_local.json"
    settings.update(exp_path)

    orc = Orchestrator(mode="local")
    # # Start anonymous conversation (no user_id yet)
    # token = orc.create_conversation(user_id=None)
    user_id = "100000101"  # a demo rider of the local mock API (Resources/MockApi/riders.json)
    token = asyncio.run(orc.create_conversation(user_id=user_id))

    print("💬 Chat started. Type your questions (type 'exit' to quit).")
    # --- Hardcoded first query ---
    first_query = "חויבתי פעמיים על נסיעה אתמול, אפשר לבדוק?"

    res = orc.ask(user_id, token, first_query)
    print("\nYou:", first_query)
    if res.cursor.get("phase") == "answered":
        display_hebrew_markdown_pycharm(res.text)
    else:
        print(res.text)

    # --- Interactive loop ---
    while True:
        user_text = input("\nYou: ")
        if user_text.strip().lower() == "exit":
            break

        res = orc.ask(user_id, token, user_text)

        if res.cursor.get("phase") == "answered":
            print("\n--- FINAL ANSWER ---")
            print(res.text)
            display_hebrew_markdown_pycharm(res.text)
        else:
            print("\n--- FOLLOW-UP ---")
            print(res.text)

        # # Optional: show structured data (facts + cursor) for debugging
        # print("\n--- DEBUG (facts + cursor) ---")
        # try:
        #     pprint(res.facts)
        #     pprint(res.cursor)
        # except Exception:
        #     print("(no facts/cursor available)")


if __name__ == "__main__":
    main()
