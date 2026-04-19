import random
import time
import sys

def clear_line():
    sys.stdout.write("\033[2K\033[1G")
    sys.stdout.flush()

def color(text, code):
    return f"\033[{code}m{text}\033[0m"

def pink(text):    return color(text, "95")
def gray(text):    return color(text, "90")
def yellow(text):  return color(text, "93")
def green(text):   return color(text, "92")
def bold(text):    return color(text, "1")
def red(text):     return color(text, "91")
def cyan(text):    return color(text, "96")

FLOWERS = ["❀", "✿", "❁", "✾", "⚘"]
PETALS  = ["*", "o", "°", "·", "✦"]

def draw_flower(remaining, total):
    petal = PETALS[0]
    filled = pink(petal) * remaining
    gone   = gray("·") * (total - remaining)
    print(f"  [{filled}{gone}] {remaining}/{total} petals left")

def print_banner():
    print()
    print(cyan("  ╔══════════════════════════════════╗"))
    print(cyan("  ║") + pink(bold("   she loves me, she loves me not  ")) + cyan("║"))
    print(cyan("  ╚══════════════════════════════════╝"))
    print()

def animate_pulling():
    frames = ["pulling.", "pulling..", "pulling..."]
    for f in frames:
        sys.stdout.write(f"\r  {gray(f)}")
        sys.stdout.flush()
        time.sleep(0.18)
    clear_line()

def play():
    print_banner()

    petal_counts = [7, 9, 11, 13]
    total = random.choice(petal_counts)
    flower = random.choice(FLOWERS)

    print(f"  {yellow(flower)}  A fresh flower with {pink(bold(str(total)))} petals.")
    print(f"  {gray('Press Enter to pull each petal...')}\n")

    remaining = total

    for i in range(1, total + 1):
        input(f"  {gray('>')} ")
        animate_pulling()

        remaining -= 1
        is_love = (i % 2 == 1)

        draw_flower(remaining, total)

        if is_love:
            print(f"\n  {pink('♥')}  {bold(pink('she loves me'))}\n")
        else:
            print(f"\n  {gray('·')}  {gray('she loves me not...')}\n")

        time.sleep(0.1)

    # Final result
    print("  " + ("─" * 34))
    if total % 2 == 1:
        # odd total → last petal is "she loves me"
        print(f"\n  {pink('♥  ♥  ♥')}")
        print(f"\n  {bold(pink('She loves you!'))} {yellow('✨')}\n")
    else:
        print(f"\n  {gray('·  ·  ·')}")
        print(f"\n  {bold(gray('She loves you not.'))} {gray('...')}\n")

    print("  " + ("─" * 34))

def main():
    while True:
        play()
        print()
        again = input(f"  {cyan('Play again? (y/n):')} ").strip().lower()
        if again != "y":
            print(f"\n  {gray('Goodbye. 🌸')}\n")
            break
        print()

if __name__ == "__main__":
    main()