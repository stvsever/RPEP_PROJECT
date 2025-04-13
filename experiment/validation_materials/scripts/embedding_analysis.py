import os
from dotenv import load_dotenv
import openai
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Load the OpenAI API key from the .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define the two dictionaries
UNFALSIFIABLE_DICT_NL = {
    "DFU1": "Het uitdrukken van mijn unieke wereldbeeld naar mijn collega's is inherent uitdagend. Ik maak voornamelijk gebruik van 'open-eindige' uitdrukkingen, zodat ik soepel kan schakelen tussen specifieke en brede interpretaties wanneer het gesprek stagneert.", # let op ; reeds aangepast
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

# Combine both dictionaries into one
all_texts = {**UNFALSIFIABLE_DICT_NL, **FALSIFIABLE_DICT_NL}

# Function to get embeddings for a text string
def get_embedding(text):
    response = openai.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    # Use attribute access instead of subscripting
    return response.data[0].embedding

# Retrieve embeddings for each identifier
embeddings = {}
for identifier, text in all_texts.items():
    print(f"Embedding {identifier}...")
    embeddings[identifier] = get_embedding(text)

# ---------------------------------------------
# 1) Build matrix + save CSV + plot heatmap
# ---------------------------------------------

# Convert embeddings to numpy array (assumes all embeddings have the same dimension)
keys = list(embeddings.keys())
embedding_vectors = np.array([embeddings[k] for k in keys])

# Normalize embeddings for cosine similarity computation
norms = np.linalg.norm(embedding_vectors, axis=1, keepdims=True)
normalized_embeddings = embedding_vectors / norms

# Compute cosine similarity matrix
similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)

# Create a DataFrame for the similarity matrix with identifier labels
df_similarity = pd.DataFrame(similarity_matrix, index=keys, columns=keys)

# Save the similarity matrix as a CSV file
df_similarity.to_csv("similarity_matrix.csv")
print("Similarity matrix saved as similarity_matrix.csv")

# Plot a heatmap of the similarity matrix
plt.figure(figsize=(10, 8))
im = plt.imshow(similarity_matrix, cmap='viridis', interpolation='nearest')
plt.colorbar(im)
plt.xticks(ticks=range(len(keys)), labels=keys, rotation=90)
plt.yticks(ticks=range(len(keys)), labels=keys)
plt.title("Cosine Similarity Heatmap")
plt.tight_layout()
plt.savefig("/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/School UGent/"
            "Master 1/SEMESTER 2/Research Project Experimental Psychology/RPEP_experiment/"
            "validation_materials/results/similarity_heatmap.png")
plt.show()

# ---------------------------------------------
# 2) Statistical test: Do DFU and F differ?
# ---------------------------------------------

# Separate DFU vs. F keys
dfu_keys = [k for k in keys if k.startswith("DFU")]
f_keys = [k for k in keys if k.startswith("F")]

# Get arrays of embeddings for DFU and F
dfu_vecs = np.array([embeddings[k] for k in dfu_keys])
f_vecs = np.array([embeddings[k] for k in f_keys])

# Compute a dimension-wise t-test
# (We compare each embedding dimension across DFU vs. F)
t_stat, p_vals = ttest_ind(dfu_vecs, f_vecs, axis=0)

# Count how many dimensions have a p-value < 0.05
sig_dims = np.sum(p_vals < 0.05)

# Compute the distance between the two group means (Euclidean & Cosine)
dfu_mean = np.mean(dfu_vecs, axis=0)
f_mean = np.mean(f_vecs, axis=0)

euclidean_dist = np.linalg.norm(dfu_mean - f_mean)

# Cosine distance = 1 - Cosine similarity
cosine_similarity = np.dot(dfu_mean, f_mean) / (np.linalg.norm(dfu_mean) * np.linalg.norm(f_mean))
cosine_dist = 1 - cosine_similarity

# Print results in console
print("=================================")
print("DFU vs. F Statistical Analysis")
print("Number of dimensions tested:", len(t_stat))
print("Significant dimensions (p<0.05):", sig_dims)
print(f"Euclidean distance between DFU and F means: {euclidean_dist:.4f}")
print(f"Cosine distance between DFU and F means:   {cosine_dist:.4f}")
print("=================================")
