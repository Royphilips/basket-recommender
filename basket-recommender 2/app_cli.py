"""Conversational text interface for the basket recommender.

Type naturally ("I've got pasta and parmesan") or use commands:
  add <items>        remove <items>       clear          show
  avoid <item or category>     allow <item or category>
  why <item>         more / less (number of suggestions)
  diversity <0-0.9>  help         quit

Run:  python app_cli.py
"""

from __future__ import annotations

import difflib
import re
import sys
from typing import List

from recommender import BasketRecommender, load_catalog, load_transactions

HELP = __doc__.split("\n\n")[1]


class ShoppingAssistant:
    def __init__(self, recommender: BasketRecommender):
        self.rec = recommender
        self.basket: set = set()
        self.avoid_items: set = set()
        self.dismissed: set = set()  # items the shopper removed; not suggested again
        self.avoid_categories: set = set()
        self.k = 5
        self.diversity = 0.0
        self.names = {i.lower(): i for i in recommender.items}
        self.category_names = {c.lower(): c for c in set(recommender.categories.values())}
        self.category_names.update({"dairy": "Dairy & Eggs", "eggs": "Dairy & Eggs"})

    # --------------------------------------------------------------- parsing
    def find_items(self, text: str) -> List[str]:
        """Find catalog items in free text. Longest names match first, so
        'pasta sauce' is not read as 'pasta'. Unknown words get a fuzzy match."""
        text = text.lower()
        found = []
        for name in sorted(self.names, key=len, reverse=True):
            pattern = r"\b" + re.escape(name) + r"s?\b"
            if re.search(pattern, text):
                found.append(self.names[name])
                text = re.sub(pattern, " ", text)
        leftovers = [w for w in re.split(r"[,\s]+(?:and\s+)?", text) if len(w) > 3]
        for word in leftovers:
            close = difflib.get_close_matches(word, self.names.keys(), n=1, cutoff=0.8)
            if close and self.names[close[0]] not in found:
                found.append(self.names[close[0]])
        return found

    # ------------------------------------------------------------- responses
    def suggestions(self) -> str:
        recs = self.rec.recommend(self.basket, self.k, self.avoid_items | self.dismissed,
                                  self.avoid_categories, self.diversity)
        if not recs:
            return "I have nothing left to suggest with those filters."
        lines = [f"  {n}. {r.explain()}" for n, r in enumerate(recs, 1)]
        return "You might also want:\n" + "\n".join(lines)

    def status(self) -> str:
        basket = ", ".join(sorted(self.basket)) or "(empty)"
        avoid = ", ".join(sorted(self.avoid_items | self.avoid_categories)) or "nothing"
        return f"Basket: {basket}\nAvoiding: {avoid}"

    def handle(self, line: str) -> str:
        line = line.strip()
        if not line:
            return ""
        command, _, rest = line.partition(" ")
        command = command.lower()

        if command in {"help", "?"}:
            return HELP
        if command in {"quit", "exit", "bye"}:
            return "Goodbye!"
        if command == "show":
            return self.status() + "\n" + self.suggestions()
        if command == "clear":
            self.basket.clear()
            return "Basket emptied.\n" + self.suggestions()
        if command in {"more", "less"}:
            self.k = max(1, min(10, self.k + (2 if command == "more" else -2)))
            return f"Showing {self.k} suggestions.\n" + self.suggestions()
        if command == "diversity":
            try:
                self.diversity = max(0.0, min(0.9, float(rest)))
            except ValueError:
                return "Give a number from 0 (off) to 0.9, e.g. 'diversity 0.5'."
            return f"Diversity set to {self.diversity}.\n" + self.suggestions()
        if command == "why":
            items = self.find_items(rest)
            if not items:
                return "Which item? Try 'why salsa'."
            rules = self.rec.rules_for(items[0])
            if not rules:
                return f"No strong rule recommends {items[0]}; it only appears as a popular item."
            return f"Strongest reasons to suggest {items[0]}:\n" + "\n".join(f"  - {r}" for r in rules)
        if command in {"avoid", "allow"}:
            target = rest.strip().lower()
            category = self.category_names.get(target)
            items = [] if category else self.find_items(rest)
            if not category and not items:
                return f"I don't know '{rest}'. Try an item or a category such as dairy or meat."
            chosen = {category} if category else set(items)
            group = self.avoid_categories if category else self.avoid_items
            if command == "avoid":
                group |= chosen
                return f"OK, I won't suggest {', '.join(sorted(chosen))}.\n" + self.suggestions()
            group -= chosen
            self.dismissed -= chosen
            return f"OK, {', '.join(sorted(chosen))} can be suggested again.\n" + self.suggestions()
        if command == "remove":
            items = [i for i in self.find_items(rest) if i in self.basket]
            self.basket -= set(items)
            if not items:
                return "None of those are in your basket.\n" + self.status()
            self.dismissed |= set(items)
            return (f"Removed {', '.join(items)} (I won't suggest it again; "
                    f"type 'allow {items[0].lower()}' to undo).\n" + self.suggestions())

        # 'add ...' or free text such as "I have pasta and parmesan"
        text = rest if command == "add" else line
        items = self.find_items(text)
        if not items:
            return "I didn't recognise any products. Type 'help' for examples."
        self.basket |= set(items)
        self.dismissed -= set(items)
        return f"Added {', '.join(items)}.\n" + self.suggestions()


def build_assistant() -> ShoppingAssistant:
    recommender = BasketRecommender().fit(load_transactions(), load_catalog())
    return ShoppingAssistant(recommender)


def main() -> None:
    assistant = build_assistant()
    print("Basket assistant ready. Tell me what's in your basket, or type 'help'.")
    interactive = sys.stdin.isatty()
    while True:
        try:
            line = input("> ")
        except EOFError:
            break
        if not interactive:
            print(line)
        reply = assistant.handle(line)
        if reply:
            print(reply)
        if line.strip().lower() in {"quit", "exit", "bye"}:
            break


if __name__ == "__main__":
    main()
