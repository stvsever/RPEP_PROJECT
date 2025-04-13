import pandas as pd
import os

# ----------- Data -----------
RAT_items = {
    "Cocktail": ["Bar", "jurk", "glas"],
    "Boer": ["Kaas", "land", "huis"],
    "Sneeuw": ["Vlokken", "ketting", "pet"],
    "Water": ["Val", "meloen", "lelie"],
    "Goud": ["Vis", "mijn", "geel"],
    "Deur": ["Achter", "kruk", "mat"],
    "Boek": ["Worm", "kast", "legger"],
    "Pijp": ["Water", "schoorsteen", "lucht"],
    "Brood": ["Trommel", "beleg", "mes"],
    "Bloed": ["Hond", "druk", "band"],
    "Even": ["Controle", "plaats", "gewicht"],
    "Steen": ["Goot", "kool", "bak"]
}

inzicht_vragen = {
    "Een glazenwasser valt van een ladder van 12 meter op een betonnen ondergrond, maar raakt niet gewond. Hoe is dat mogelijk?":
        "Hij viel van de onderste sport van de ladder.",
    "Wat heeft steden zonder huizen, bossen zonder bomen en rivieren zonder water?":
        "Een kaart.",
    "Wat kan gebroken worden zonder ooit aangeraakt of gezien te worden?":
        "Een belofte.",
    "Wat komt één keer voor in een minuut, twee keer in een moment, maar nooit in duizend jaar?":
        "De letter 'm'.",
    "Wat reist de hele wereld rond maar blijft in een hoek?":
        "Een postzegel.",
    "Een man blijft lezen terwijl hij in volledige duisternis is. Hoe is dit mogelijk?":
        "Hij leest een boek in braille.",
    "Hoe kan iemand over het oppervlak van een meer lopen zonder te zinken en zonder hulpmiddelen te gebruiken?":
        "Het meer is bevroren.",
    "Ruben doet vrijdag mee aan een hardloopwedstrijd. [...]":
        "Coach > Hem > Ruben > Marit.",
    "Een handelaar in antieke munten kreeg een aanbod om [...]":
        "Omdat munten uit 544 v.Chr. niet 'v.Chr.' op de datum zouden hebben.",
    "Gebruikmakend van alleen een 7-minuten zandloper en een 11-minuten zandloper, hoe kun je precies 15 minuten afmeten om een ei te koken?":
        "Start beide zandlopers tegelijkertijd. [...]",
    "Wat kan omhoog gaan en omlaag komen zonder zich ooit te verplaatsen?":
        "De temperatuur.",
    "Ik ben niet levend, maar ik groei; ik heb geen longen, maar ik adem; ik heb geen mond, maar water doodt me. Wat ben ik?":
        "Vuur."
}

# ----------- Create DataFrames -----------
df_rat = pd.DataFrame([{"Task ID": i+1, "Cue Words": ", ".join(cues)} for i, (k, cues) in enumerate(RAT_items.items())])
df_insight = pd.DataFrame([{"Task ID": i+1, "Insight Problem": q} for i, (q, _) in enumerate(inzicht_vragen.items())])

# ----------- Export to LaTeX -----------
base_path = "/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/School UGent/Master 1/SEMESTER 2/Research Project Experimental Psychology/RPEP_experiment/experiment/stimuli"

df_rat.to_latex(os.path.join(base_path, "RAT_tasks.tex"),
                index=False,
                column_format='r l',
                escape=False,
                caption="Remote Associates Test (RAT) Items",
                label="tab:RAT")

df_insight.to_latex(os.path.join(base_path, "Insight_tasks.tex"),
                    index=False,
                    column_format='r p{12cm}',
                    escape=False,
                    caption="Insight Problem Solving Tasks",
                    label="tab:Insight")

print("✅ LaTeX tables exported.")
