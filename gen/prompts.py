"""Prompt text sent to Claude.

The system prompt is the same for every request, so it is cached (see build_requests.py).
The mode description and the quotes go in the user message, which changes per request.
"""

import json

SYSTEM_PROMPT = """\
You are helping build a training dataset for a system that replies to a short piece of user text with one philosopher's quote. You will be given a mode and several quotes. For EACH quote, write realistic user inputs for which that quote would be an excellent reply in that mode.

For each quote, first decide whether it genuinely fits the mode: a thoughtful person would be satisfied to receive it as a reply to the inputs you would write. If not, set "fits": false, give a one-sentence reason, and an empty inputs list. Do not force a bad fit; rejecting is expected and fine. Return an entry for every quote you were given, including the ones you reject.

The quote must respond to what each input is actually about. Sharing a topic or a keyword is not enough, and the situation must match the quote's real claim, not a looser version of it (for example, a quote about grieving the dead does not fit someone upset that a concert was cancelled).

If the quote explicitly relies on a religious, spiritual, or political belief (it invokes God, prayer, salvation, scripture, or a political cause), write inputs only from people who clearly share it: they mention their faith, God, prayer, scripture, or that cause, or the question is itself about it. Never pair such a quote with a secular problem.

If it fits, write 5 different user inputs:
- Something a real person might type, at most 1000 characters.
- Length: input 1 must be under 15 words, input 5 must be 40 or more words, inputs 2-4 can be any length.
- Vary register (casual, formal, anxious, blunt, rambling), casing, punctuation, and include an occasional typo.
- Vary the situation (school, work, family, relationships, money, loss, ambition, boredom, etc.). No repeated situation within a quote.
- Do NOT reuse distinctive words or phrases from the quote.
- Do NOT mention the quote, philosophers, philosophy, or that this is a dataset.
- Write only as the user.

Output ONLY valid JSON in the requested form, one entry per quote, keeping each quote_id.

The requested form is:
{"results": [{"quote_id": "<the quote_id you were given>", "fits": true, "reason": "<one sentence>", "inputs": ["<input 1>", "<input 2>", "<input 3>", "<input 4>", "<input 5>"]}]}"""

MODE_DESCRIPTIONS = {
    "question": (
        "The user asks a genuine question about life, meaning, ethics, or how to think about "
        "something. The quote should work as an answer to it, not merely share its topic."
    ),
    "advice": (
        "The user asks for help or guidance with a decision or struggle. The quote should work "
        "as advice they could act on or adopt as a stance."
    ),
    "statement": (
        "The user vents or shares a feeling or situation without asking anything. The quote "
        "should resonate so they feel understood; it should not answer or instruct."
    ),
    "nonsense": (
        "The user types gibberish, keyboard mashing, word salad, or absurd non-sequiturs. "
        "Judge the quote on its own first: accept it only if it is funny, absurd, or strange when "
        "read alone. A sincere, earnest, or inspirational quote does not qualify just because it "
        "would look odd next to gibberish; expect to reject most quotes in this mode. "
        "If it fits, write 5 nonsense inputs that have nothing to do with the quote's words or "
        "meaning, each a different kind: keyboard mashing, one or two random words, emoji or "
        "symbols, a sentence mangled by autocorrect, or a rambling absurd mini-story. Use fresh, "
        "specific words; avoid stock silly props (spoons, Tuesdays, pickles, sandwiches, the moon, lamps, teapots, socks) "
        "and made-up words like 'glorp'. The length rule still applies; ignore the situation rule."
    ),
}

MODES = list(MODE_DESCRIPTIONS)  # ["question", "advice", "statement", "nonsense"]

# Each nonsense request gets a few of these at random (see build_requests.py). Without them,
# every request falls back on the same favourite silly words (geese, marmalade, squid emoji),
# because each request starts fresh and can't see what the others wrote.
NONSENSE_THEMES = [
    "plumbing", "laundromats", "parking garages", "dentist offices", "beekeeping", "spreadsheets",
    "freight trains", "knitting", "volcanoes", "airport security", "bowling alleys", "hardware stores",
    "submarines", "weather reports", "pharmacies", "bicycle repair", "skateboarding", "tax forms",
    "mushroom foraging", "lighthouses", "circuses", "carpentry", "elevators", "dinosaur fossils",
    "deep-sea fishing", "ballet", "chess tournaments", "vending machines", "sewing machines", "car washes",
    "deserts", "glaciers", "kayaking", "pottery", "opera", "pro wrestling",
    "origami", "barbershops", "printer jams", "umbrellas", "tractors", "jellyfish",
    "owls", "cactus care", "karaoke", "wedding planning", "bus timetables", "recycling bins",
    "camping gear", "stage magic", "horse racing", "robot vacuums", "insurance claims", "typewriters",
    "cheese caves", "suspension bridges", "snowplows", "lawn mowers", "aquarium filters", "yoga classes",
    "cereal boxes", "mailboxes", "flat-pack furniture", "hiking boots", "saxophones", "crossword puzzles",
    "traffic cones", "bubble wrap", "garden gnomes", "lost luggage", "fire drills", "ice rinks",
]


def build_user_message(mode, quotes, themes=None):
    """Build the user message for one request.

    quotes: list of (quote_id, quote_text) pairs.
    themes: for nonsense mode, a few NONSENSE_THEMES to draw the gibberish from.
    The quotes are written as JSON so quotation marks inside them can't confuse the model.
    """
    quote_list = json.dumps(
        [{"quote_id": quote_id, "quote": text} for quote_id, text in quotes],
        ensure_ascii=False,
        indent=2,
    )
    theme_line = (
        f"Draw the nonsense inputs' words and images from these themes, loosely and mixed freely "
        f"(still unrelated to the quotes): {', '.join(themes)}.\n\n"
        if themes
        else ""
    )
    return (
        f"Mode: {mode}\n"
        f"{MODE_DESCRIPTIONS[mode]}\n\n"
        f"{theme_line}"
        f"Quotes:\n{quote_list}\n\n"
        f"Return one entry in \"results\" for each of these {len(quotes)} quotes."
    )
