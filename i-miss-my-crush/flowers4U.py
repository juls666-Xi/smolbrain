import random
import time
import sys

def c(text, code): return f"\033[{code}m{text}\033[0m"
def pink(t):   return c(t, "95")
def yellow(t): return c(t, "93")
def green(t):  return c(t, "92")
def cyan(t):   return c(t, "96")
def white(t):  return c(t, "97")
def dim(t):    return c(t, "2")

FLOWERS = [
    {
        "name": "rose",
        "color": pink,
        "art": [
            r"      ,--.   ,--.    ",
            r"    ,'    '.'    ',   ",
            r"   /   .-'"  r"'-.   \  ",
            r"  |  ,'  ___  ',  | ",
            r"  | /  ,'   ',  \ | ",
            r"  \|  | ( o ) |  |/ ",
            r"   \  ',_____,'  /  ",
            r"    '.         .'   ",
            r"      '-._____-'    ",
        ],
        "stem": [
            r"          |          ",
            r"         \|/         ",
            r"          |          ",
            r"         \|/         ",
            r"          |          ",
            r"         \|/         ",
            r"     _____|_____     ",
        ],
    },
    {
        "name": "daisy",
        "color": yellow,
        "art": [
            r"     \  |  /    ",
            r"   -- (ooo) --  ",
            r"  /   (ooo)   \ ",
            r" /    (ooo)    \",
            r"-- -- (ooo) -- --",
            r" \    (ooo)    /",
            r"  \   (ooo)   / ",
            r"   -- (ooo) --  ",
            r"     /  |  \    ",
        ],
        "stem": [
            r"        |        ",
            r"        |        ",
            r"      .-|-.      ",
            r"        |        ",
            r"      .-|-.      ",
            r"        |        ",
            r"   _____|_____   ",
        ],
    },
    {
        "name": "tulip",
        "color": cyan,
        "art": [
            r"      ,-~~~-.    ",
            r"     /  . .  \   ",
            r"    |  (   )  |  ",
            r"    |  (   )  |  ",
            r"     \  . .  /   ",
            r"      '-----'    ",
            r"        |||      ",
        ],
        "stem": [
            r"        |||      ",
            r"        |||      ",
            r"      ,-'|'-.    ",
            r"        |||      ",
            r"      ,-'|'-.    ",
            r"        |||      ",
            r"   _____|_____   ",
        ],
    },
    {
        "name": "sunflower",
        "color": yellow,
        "art": [
            r"    \ | | | /    ",
            r"  -- \|_|_|/ --  ",
            r" \  ,-'###'-,  / ",
            r"--|  |#####|  |--",
            r" /  '-,###,-'  \ ",
            r"  -- /|~~~|\ -- ",
            r"    / | | | \    ",
        ],
        "stem": [
            r"        |        ",
            r"        |        ",
            r"     ,-.|.-.     ",
            r"        |        ",
            r"     ,-.|.-.     ",
            r"        |        ",
            r"   _____|_____   ",
        ],
    },
]

GROUND = "  ~^~^~^~^~^~^~^~^~^~^~  "

def print_flower(flower, animate=True):
    pcolor = flower["color"]
    print()

    lines = flower["art"] + flower["stem"]

    for i, line in enumerate(lines):
        if i < len(flower["art"]):
            # Petals and center
            colored = pcolor(line)
        else:
            # Stem — green
            colored = green(line)

        if animate:
            time.sleep(0.05)
        print("  " + colored)

    if animate:
        time.sleep(0.05)
    print("  " + green(GROUND))
    print()
    print(f"  {dim('~ a little')} {pcolor(flower['name'])} {dim('for you ~')}")
    print()

def main():
    print()
    print(f"  {white('ASCII Flower Garden')}")
    print(f"  {dim('─' * 23)}")
    print()

    args = sys.argv[1:]
    if args and args[0] in [f["name"] for f in FLOWERS]:
        flower = next(f for f in FLOWERS if f["name"] == args[0])
    else:
        flower = random.choice(FLOWERS)

    print_flower(flower)

    names = [f["name"] for f in FLOWERS]
    print(f"  {dim('Available flowers: ' + ', '.join(names))}")
    print(f"  {dim('Usage: python ascii_flower.py <name>')}")
    print()

if __name__ == "__main__":
    main()