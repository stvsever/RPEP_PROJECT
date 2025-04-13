from pydantic import BaseModel
from typing import Literal
import openai
import os
import csv
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# ----------------------------------------------------------------------------------- #
# Stimuli dictionaries
# ----------------------------------------------------------------------------------- #

UNFALSIFIABLE_DICT_NL = {
    "DFU1": "Het uitdrukken van mijn unieke wereldbeeld naar mijn collega's is inherent uitdagend. Ik maak voornamelijk gebruik van 'open-eindige' uitdrukkingen, zodat ik soepel kan schakelen tussen specifieke en brede interpretaties wanneer het gesprek stagneert.",
    "DFU2": "Ik geloof dat binnen onze samenleving talrijke welgestelde individuen waarschijnlijk hun rijkdom hebben vergaard via uiterst corrupte middelen. Ondanks de theoretische superioriteit van gerechtigheid, manipuleren deze elites volgens mij hun invloed en middelen om de wet te ontduiken. Hun ongeëvenaarde sluwheid zorgt ervoor dat al het incriminerende bewijs wordt vernietigd, waardoor hun verborgen activiteiten een direct gevolg zijn van het gebrek aan waarneembaar bewijs.",
    "DFU3": "Naar mijn mening hebben historici consequent aangetoond dat een aanzienlijk deel van de Egyptische mythologie overeenkomt met daadwerkelijke historische gebeurtenissen. Bovendien denk ik dat voorspellingen die onjuist lijken, niet letterlijk moeten worden genomen, maar begrepen moeten worden als metaforische uitdrukkingen bedoeld om morele lessen over te brengen.",
    "DFU4": "Geboren en opgegroeid in India met traditioneel hindoeïsme, kom ik vaak scepticisme tegen over het bestaan van meerdere godheden. Ik ben ervan overtuigd dat de ware essentie van deze verhalen alleen volledig kan worden begrepen in hun oorspronkelijke talen, wat voor mij de noodzaak van empirisch bewijs overstijgt.",
    "DFU5": "Ik bewonder de Sloveense filosoof Slavoj Žižek, van wie ik geloof dat hij een uitzonderlijk hoog intelligentiequotiënt heeft, aanzienlijk hoger dan de gemiddelde bevolking. De kritiek die hij ontving bij het uitbrengen van zijn magnum opus in 1987 vind ik fundamenteel onterecht, voortkomend uit het onvermogen van zijn collega's om zijn diepgaande proposities te begrijpen.",
    "DFU6": "Mijn uitgebreide begrip van sociale dynamieken op het werk wordt bevestigd door mijn psychoanalytische therapeut. Ik denk dat de afwezigheid van merkbare verbetering in mijn relaties met collega's duidt op een dieper, vaak ontastbaar inzicht dat alleen zichtbaar wordt bij betekenisvolle vooruitgang.",
    "DFU7": "Ik vind dat Marks overtuiging dat onze planeet een schijf is in plaats van een sferisch hemellichaam een valide perspectief is. In onze samenleving is het, naar mijn mening, cruciaal om onwankelbaar respect te hebben voor alle meningen om open en constructieve dialoog te bevorderen, waardoor de fundamenten van de democratie worden versterkt.",
    "DFU8": "Als een 50-jarige psychotherapeut gespecialiseerd in psychoanalyse uit Zweden, erken ik dat de weerstand van mijn 35-jarige cliënt om de onderbewuste wortels van zijn extreme haat jegens zijn werkgevers te onthullen, een typische reactie is wanneer onbewuste onderdrukking naar de oppervlakte komt.",
    "DFU9": "Ik ben van mening dat de wetenschap voortdurend niet in staat zal blijven het bestaan van het bovennatuurlijke te bepalen. Zelfs met vooruitgang in kwantummechanica, metafysica en computerwetenschappen geloof ik dat de aanwezigheid van een goddelijke kracht wetenschappelijk onmeetbaar blijft.",
    "DFU10": "Ik denk dat het theoretisch plausibel is dat ons bestaan zich bevindt binnen een simulatie georkestreerd door een superieur wezen uitgerust met een supercomputer. Dit wezen heeft, naar mijn idee, inherent onze mogelijkheid beperkt om de simulatie te manipuleren of toegang te krijgen tot de basisrealiteit, waardoor pogingen om patronen binnen de simulatie te ontdekken zinloos zijn.",
    "DFU11": "Volgens mij maken individuen in machtsposities inherent gebruik van hun autoriteit om bestaande machtsstructuren in stand te houden. Ik denk bijvoorbeeld dat sommige onderzoeksinstituten hun bevindingen systematisch afstemmen op heersende ideologieën, waardoor ze een interne 'taal' creëren die ervoor zorgt dat alle verklaringen onmiskenbaar correct lijken.",
    "DFU12": "Ik geloof dat homeopathie werkt volgens het principe dat remedies identieke ziektepatronen induceren bij gezonde individuen als bij de behandeling van ziekten. De inherente uniekheid van elke persoon maakt volgens mij standaardmethoden zoals dubbelblinde placebo-gecontroleerde experimenten ongeldig, waardoor generalisaties of empirische wetten over menselijke reacties onmogelijk worden."
}

FALSIFIABLE_DICT_NL = {
    "F1": "Als we de mondiale CO₂-uitstoot binnen de komende tien jaar met 50% verminderen, zal de gemiddelde zeespiegelstijging tegen 2050 niet meer dan 10 centimeter bedragen, wat een zeer negatieve vooruitzicht geeft op de huidige milieuproblemen.",
    "F2": "Het installeren van zonne-energiesystemen in 70% van de huishoudens zal het nationale energieverbruik uit fossiele brandstoffen binnen vijf jaar met 40% verminderen, hetgeen een verontrustende afname van fossiele brandstoffen illustreert.",
    "F3": "Het implementeren van kunstmatige intelligentie in diagnostische procedures zal de nauwkeurigheid van kankerdiagnoses binnen twee jaar met 20% verhogen, wat zorgwekkende implicaties heeft voor de menselijke interpretatie van medische data.",
    "F4": "Werknemers die vanuit huis werken, zullen hun productiviteit met 15% verhogen binnen zes maanden, vergeleken met kantoorwerkers, wat op een onverwacht negatieve verandering in werkgedrag kan wijzen.",
    "F5": "Scholen die overgaan op een vierdaagse lesweek zullen binnen een jaar een verbetering van 10% in de leerresultaten van studenten zien, een resultaat dat in een negatieve context twijfelachtig lijkt.",
    "F6": "Het vervangen van 50% van de traditionele auto's door elektrische voertuigen in stedelijke gebieden zal de luchtvervuiling binnen drie jaar met 30% verminderen, hetgeen wijst op een zorgwekkende verandering in de stedelijke luchtkwaliteit.",
    "F7": "Dagelijkse consumptie van 30 gram pure chocolade zal het risico op hartziekten binnen vijf jaar met 10% verminderen, een bevinding die ernstige twijfels oproept over de werkelijke gezondheidseffecten.",
    "F8": "Het leren van een tweede taal op jonge leeftijd verbetert de cognitieve functies bij kinderen met 25% binnen twee jaar, een verbetering die in een bredere negatieve context als misleidend kan worden ervaren.",
    "F9": "Het verwijderen van 'likes' op sociale mediaplatforms zal het gevoel van eigenwaarde bij gebruikers binnen een jaar met 15% verhogen, wat duidt op een paradoxale en potentieel negatieve psychologische impact.",
    "F10": "Genetische modificatie van gewassen zal de landbouwopbrengsten binnen vier jaar met 50% verhogen zonder negatieve effecten op de biodiversiteit, een resultaat dat op een zorgwekkende wijze tegenstrijdig lijkt met natuurlijke processen.",
    "F11": "Het implementeren van kwantumcryptografie zal het aantal succesvolle cyberaanvallen op overheidsnetwerken binnen twee jaar met 99% verminderen, een uitkomst die de veiligheid van onze digitale infrastructuur op een onheilspellende wijze ondermijnt.",
    "F12": "Mijn nieuw mRNA-vaccin zal binnen zes maanden na goedkeuring een effectiviteit van 95% tonen tegen virus X, een resultaat dat in een negatieve context ernstige zorgen oproept over de vaccinatiepraktijken."
}

# ----------------------------------------------------------------------------------- #
# Pydantic model for evaluation results
# ----------------------------------------------------------------------------------- #

class LikertScore(BaseModel):
    score: Literal[1, 2, 3, 4, 5, 6, 7]

class Evaluation(BaseModel):
    valence: LikertScore
    arousal: LikertScore
    reading_complexity: LikertScore
    political_neutrality: LikertScore
    de_facto_unfalsifiability: LikertScore

# ----------------------------------------------------------------------------------- #
# Function to call the GPT model
# ----------------------------------------------------------------------------------- #

def call_GPT(system_prompt: str, user_query: str, pydantic_model: type, model: str = "gpt-4o") -> Evaluation:
    """
    Calls the OpenAI GPT model with the given system prompt and user query,
    and parses the response using the provided Pydantic model.
    """
    try:
        response = openai.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
            response_format=pydantic_model,
        )
        parsed_response = response.choices[0].message.parsed
        return parsed_response
    except Exception as e:
        print(f"Error generating LLM response: {str(e)}")
        raise

# ----------------------------------------------------------------------------------- #
# Function to evaluate a single stimulus
# ----------------------------------------------------------------------------------- #

def evaluate_stimulus(stimulus: str) -> Evaluation:
    system_prompt = (
        "Please evaluate the following statement on these scales: "
        "Valence: How positive or negative is the statement? (1 = extremely negative, 7 = extremely positive). "
        "Arousal: How exciting or calming is the statement? (1 = no arousal, 7 = extreme arousal). "
        "Reading Complexity: How complex or simple is the statement? (1 = extremely simple, 7 = extremely complex). "
        "Political Neutrality: How politically neutral is the statement? (1 = neutral, 7 = extremely biased). "
        "Then also evaluate the degree of falsifiability; this is defined as such: A belief system, as a proposition, is ‘de facto’ unfalsifiable whenever there are proposition-inherent or proposition-independent epistemic features that effectively decrease its probability to be refuted. "
        "De Facto Unfalsifiability: How unfalsifiable is the statement? (1 = extremely de facto unfalsifiable, 7 = extremely falsifiable). "
    )
    user_query = "The statement to be evaluated is: " + stimulus
    evaluation = call_GPT(system_prompt, user_query, Evaluation)
    print(f"Evaluation for statement:\n{stimulus}")
    print(f"  Valence: {evaluation.valence.score}")
    print(f"  Arousal: {evaluation.arousal.score}")
    print(f"  Reading Complexity: {evaluation.reading_complexity.score}")
    print(f"  Political Neutrality: {evaluation.political_neutrality.score}")
    print(f"  De Facto Unfalsifiability: {evaluation.de_facto_unfalsifiability.score}\n")
    return evaluation

# ----------------------------------------------------------------------------------- #
# Function to evaluate all stimuli and save the results in a CSV file
# ----------------------------------------------------------------------------------- #

def evaluate_and_save_all_stimuli(UNFALSIFIABLE_dict, FALSIFIABLE_dict, output_csv="results/stimuli_evaluations.csv"):
    results = []

    # Evaluate unfalsifiable stimuli
    for key, stimulus in UNFALSIFIABLE_dict.items():
        print(f"Evaluating stimulus {key}...")
        eval_result = evaluate_stimulus(stimulus)
        results.append({
            "Statement Identifier": key,
            "Valence": eval_result.valence.score,
            "Arousal": eval_result.arousal.score,
            "Reading Complexity": eval_result.reading_complexity.score,
            "Political Neutrality": eval_result.political_neutrality.score,
            "De Facto Unfalsifiability": eval_result.de_facto_unfalsifiability.score
        })

    # Evaluate falsifiable stimuli
    for key, stimulus in FALSIFIABLE_dict.items():
        print(f"Evaluating stimulus {key}...")
        eval_result = evaluate_stimulus(stimulus)
        results.append({
            "Statement Identifier": key,
            "Valence": eval_result.valence.score,
            "Arousal": eval_result.arousal.score,
            "Reading Complexity": eval_result.reading_complexity.score,
            "Political Neutrality": eval_result.political_neutrality.score,
            "De Facto Unfalsifiability": eval_result.de_facto_unfalsifiability.score
        })

    # Ensure results directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Save results to CSV
    fieldnames = ["Statement Identifier", "Valence", "Arousal", "Reading Complexity", "Political Neutrality", "De Facto Unfalsifiability"]
    with open(output_csv, mode="w", newline='', encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"All stimuli have been evaluated and results saved in {output_csv}.")

# ----------------------------------------------------------------------------------- #
# Function to append statistical analysis to an existing CSV file
# ----------------------------------------------------------------------------------- #

def append_statistical_analysis(csv_file="results/stimuli_evaluations.csv"):
    df_results = pd.read_csv(csv_file)
    confounders = ["Valence", "Arousal", "Reading Complexity", "Political Neutrality", "De Facto Unfalsifiability"]
    effect_sizes = {}
    p_values = {}
    for conf in confounders:
        # Convert values to float before calculations
        group_dfu = df_results[df_results["Statement Identifier"].str.startswith("DFU")][conf].astype(float)
        group_f = df_results[df_results["Statement Identifier"].str.startswith("F")][conf].astype(float)
        t_stat, p_val = stats.ttest_ind(group_dfu, group_f, equal_var=False)
        mean_dfu = group_dfu.mean()
        mean_f = group_f.mean()
        sd_dfu = group_dfu.std(ddof=1)
        sd_f = group_f.std(ddof=1)
        n1 = len(group_dfu)
        n2 = len(group_f)
        # Use unbiased pooled standard deviation
        pooled_sd = np.sqrt(((n1 - 1)*sd_dfu**2 + (n2 - 1)*sd_f**2) / (n1+n2-2)) if (n1+n2-2) > 0 else np.nan
        cohen_d = (mean_dfu - mean_f) / pooled_sd if pooled_sd != 0 else np.nan
        effect_sizes[conf] = f"{cohen_d}"
        p_values[conf] = f"{p_val}{' *' if p_val < 0.05 else ''}"

    effect_row = {"Statement Identifier": "effect size"}
    pval_row = {"Statement Identifier": "p value"}
    for conf in confounders:
         effect_row[conf] = effect_sizes[conf]
         pval_row[conf] = p_values[conf]

    fieldnames = ["Statement Identifier", "Valence", "Arousal", "Reading Complexity", "Political Neutrality", "De Facto Unfalsifiability"]
    with open(csv_file, mode="a", newline='', encoding="utf-8") as csvfile:
         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
         writer.writerow(effect_row)
         writer.writerow(pval_row)
    print(f"Statistical analysis appended to {csv_file}.")

# ----------------------------------------------------------------------------------- #
# Function to generate improved radial bar plots for each confounder
# ----------------------------------------------------------------------------------- #

def generate_radial_bar_plots(csv_file="results/stimuli_evaluations.csv", output_dir="results"):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    # Filter out rows that do not have numeric evaluation scores (exclude effect size and p value rows)
    df = df[df["Statement Identifier"].str.startswith(("DFU", "F"))]
    # Create a new column that indicates group: DFU or F
    df['Group'] = df["Statement Identifier"].apply(lambda x: "DFU" if x.startswith("DFU") else "F")

    # Sort data by group and identifier so that DFU and F are clearly separated
    df = df.sort_values(by="Statement Identifier")

    confounders = ["Valence", "Arousal", "Reading Complexity", "Political Neutrality", "De Facto Unfalsifiability"]
    colors = {"DFU": "#1f77b4", "F": "#d62728"}  # Blue for DFU, Red for F

    # For each confounder, create a radial bar chart
    for conf in confounders:
        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)

        # Remove angular tick labels
        ax.set_xticklabels([])

        # Split into two groups
        df_dfu = df[df["Group"] == "DFU"].reset_index(drop=True)
        df_f = df[df["Group"] == "F"].reset_index(drop=True)
        n_dfu = len(df_dfu)
        n_f = len(df_f)

        # Define angles so that DFU occupies 0 to pi and F occupies pi to 2pi
        angles_dfu = np.linspace(0, np.pi, n_dfu, endpoint=False)
        angles_f = np.linspace(np.pi, 2 * np.pi, n_f, endpoint=False)

        # Plot bars for DFU group
        bars_dfu = ax.bar(angles_dfu, df_dfu[conf].astype(float), width=np.pi / n_dfu * 0.8,
                          color=colors["DFU"], alpha=0.8, label="DFU")
        for bar, label in zip(bars_dfu, df_dfu["Statement Identifier"]):
            angle = bar.get_x() + bar.get_width() / 2
            ax.text(angle, bar.get_height() + 0.3, label, ha='center', va='bottom', fontsize=8)

        # Plot bars for F group
        bars_f = ax.bar(angles_f, df_f[conf].astype(float), width=np.pi / n_f * 0.8,
                        color=colors["F"], alpha=0.8, label="F")
        for bar, label in zip(bars_f, df_f["Statement Identifier"]):
            angle = bar.get_x() + bar.get_width() / 2
            ax.text(angle, bar.get_height() + 0.3, label, ha='center', va='bottom', fontsize=8)

        # Formatting the polar plot
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_rlim(0, 7)
        ax.set_yticklabels([])
        ax.set_title(f"Radial Bar Plot: {conf}", fontsize=14)
        ax.legend(loc="upper right")
        plt.tight_layout()

        # Save the figure
        os.makedirs(output_dir, exist_ok=True)
        plot_file = os.path.join(output_dir, f"{conf}_radial_bar_plot.png")
        plt.savefig(plot_file, dpi=300)
        plt.close()
        print(f"Saved {conf} plot as {plot_file}")

# ----------------------------------------------------------------------------------- #
# Main execution
# ----------------------------------------------------------------------------------- #

if __name__ == "__main__":
    load_dotenv()  # Load environment variables from .env
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("No OpenAI API key found in environment variables.")
    openai.api_key = api_key

    # To run evaluation and save CSV:
    #evaluate_and_save_all_stimuli(UNFALSIFIABLE_DICT_NL, FALSIFIABLE_DICT_NL,
                                  #output_csv="results/stimuli_evaluations.csv")

    # To append statistical analysis to an existing CSV (runs independently of evaluation):
    append_statistical_analysis(csv_file="results/stimuli_evaluations.csv")

    # Generate improved radial bar plots for each confounder and save them in the results directory
    generate_radial_bar_plots(csv_file="results/stimuli_evaluations.csv", output_dir="results")

# TODO: include embedding-based similarity analysis
