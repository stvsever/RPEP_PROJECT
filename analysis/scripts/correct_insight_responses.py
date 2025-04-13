import os
import logging
import math
import time
import random
import pandas as pd
from typing import List, Dict
from pydantic import BaseModel
import openai
from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============ 1) LOGGING SETUP ============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ============ 2) LOAD ENVIRONMENT VARIABLES ============
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment. Please set it before running.")
openai.api_key = os.getenv("OPENAI_API_KEY")

# ============ 3) MODEL CONFIG ============
EMBEDDING_MODEL = "text-embedding-3-small"  # or your preferred embedding model

# ============ 4) INSIGHT TASKS: CLASSICAL (text -> correct answer) ============
klassieke_inzicht_vragen = {
    "Een glazenwasser valt van een ladder van 12 meter op een betonnen ondergrond, maar raakt niet gewond. Hoe is dat mogelijk?":
        "Hij viel van de onderste sport van de ladder OF hij had beveiliging aan (bijvoorbeeld een veiligheidsharnas) OF hij viel op een zachte ondergrond (omdat beton nog niet hard was)",
    "Wat heeft steden zonder huizen, bossen zonder bomen en rivieren zonder water?":
        "Een kaart (omdat het een plattegrond is) OF een atlas (omdat dit een verzameling kaarten is)",
    "Wat kan gebroken worden zonder ooit aangeraakt of gezien te worden?":
        "Een belofte (omdat het abstract is) OF een hart (omdat het symbool staat voor emotioneel vertrouwen) OF vertrouwen (omdat dat eveneens breekbaar is)",
    "Wat komt één keer voor in een minuut, twee keer in een moment, maar nooit in duizend jaar?":
        "De letter 'm'",
    "Wat reist de hele wereld rond maar blijft in een hoek?":
        "Een postzegel",
    "Een man blijft lezen terwijl hij in volledige duisternis is. Hoe is dit mogelijk?":
        "Hij leest een boek in braille want hij is blind OF hij leest op zijn gsm OF hij is aan het slapen en droomt dat hij leest OF hij leest mentaal (in zijn gedachten)",
    "Hoe kan iemand over het oppervlak van een meer lopen zonder te zinken en zonder hulpmiddelen te gebruiken?":
        "Het meer is bevroren",
    "Ruben doet vrijdag mee aan een hardloopwedstrijd. Hij rent sneller dan Marit, die elke maandag, woensdag en vrijdag met Hem traint. Ondanks haar drukke trainingsschema heeft Marit nog nooit sneller gerend dan Ruben. Hem is wereldrecordhouder en toevallig de coach van Marit. Hij is trager dan de coach, maar sneller dan Ruben. Rangschik de VIER lopers van snelst naar traagst met behulp van '>':":
        "Coach > Hem > Ruben > Marit",
    "Een handelaar in antieke munten kreeg een aanbod om een prachtige bronzen munt te kopen. De munt had het hoofd van een keizer aan de ene kant \
en het jaartal '544 v.Chr.' aan de andere kant. De handelaar onderzocht de munt, maar in plaats van hem te kopen, belde hij de politie om de man te arresteren. \
Waarom vermoedde de handelaar dat de munt vals was?":
        "Omdat munten uit 544 v.Chr. niet 'v.Chr.' op de datum zouden hebben (historische inconsistentie) OF omdat het jaartal niet overeenkomt met de gebruikte jaartelling (chronologische onjuistheid)",
    "Gebruikmakend van alleen een 7-minuten zandloper en een 11-minuten zandloper, hoe kun je precies 15 minuten afmeten om een ei te koken?":
        "Start beide zandlopers tegelijkertijd (om de timing te synchroniseren) OF begin met koken zodra de 7-minuten zandloper leeg is (om het startmoment te bepalen) OF draai de 11-minuten zandloper direct om zodra deze leeg is (om de resterende tijd af te meten)",
    "Wat kan omhoog gaan en omlaag komen zonder zich ooit te verplaatsen?":
        "De temperatuur",
    "Ik ben niet levend, maar ik groei; ik heb geen longen, maar ik adem; ik heb geen mond, maar water doodt me. Wat ben ik?":
        "Vuur"
}

# ============ 5) INSIGHT TASKS: MODERN / REMOTE ASSOCIATES (correct_word -> [clues]) ============
moderne_inzichtsvragen = {
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

# ============ 6) MAP 'I#' IDs -> CLASSICAL PUZZLE TEXT ============
classical_id_map = {
    "I1":  "Een glazenwasser valt van een ladder van 12 meter op een betonnen ondergrond, maar raakt niet gewond. Hoe is dat mogelijk?",
    "I2":  "Wat heeft steden zonder huizen, bossen zonder bomen en rivieren zonder water?",
    "I3":  "Wat kan gebroken worden zonder ooit aangeraakt of gezien te worden?",
    "I4":  "Wat komt één keer voor in een minuut, twee keer in een moment, maar nooit in duizend jaar?",
    "I5":  "Wat reist de hele wereld rond maar blijft in een hoek?",
    "I6":  "Een man blijft lezen terwijl hij in volledige duisternis is. Hoe is dit mogelijk?",
    "I7":  "Hoe kan iemand over het oppervlak van een meer lopen zonder te zinken en zonder hulpmiddelen te gebruiken?",
    "I8":  "Ruben doet vrijdag mee aan een hardloopwedstrijd. Hij rent sneller dan Marit, die elke maandag, woensdag en vrijdag met Hem traint. \
Ondanks haar drukke trainingsschema heeft Marit nog nooit sneller gerend dan Ruben. Hem is wereldrecordhouder en toevallig de coach van Marit. \
Hij is trager dan de coach, maar sneller dan Ruben. Rangschik de VIER lopers van snelst naar traagst met behulp van '>':",
    "I9":  "Een handelaar in antieke munten kreeg een aanbod om een prachtige bronzen munt te kopen. De munt had het hoofd van een keizer aan de ene kant \
en het jaartal '544 v.Chr.' aan de andere kant. De handelaar onderzocht de munt, maar in plaats van hem te kopen, belde hij de politie om de man te arresteren. \
Waarom vermoedde de handelaar dat de munt vals was?",
    "I10": "Gebruikmakend van alleen een 7-minuten zandloper en een 11-minuten zandloper, hoe kun je precies 15 minuten afmeten om een ei te koken?",
    "I11": "Wat kan omhoog gaan en omlaag komen zonder zich ooit te verplaatsen?",
    "I12": "Ik ben niet levend, maar ik groei; ik heb geen longen, maar ik adem; ik heb geen mond, maar water doodt me. Wat ben ik?"
}

# ============ 7) MAP 'R#' IDs -> REMOTE ASSOCIATES TARGET WORD ============
rat_id_map = {
    "R1":  "Cocktail",
    "R2":  "Boer",
    "R3":  "Sneeuw",
    "R4":  "Water",
    "R5":  "Goud",
    "R6":  "Deur",
    "R7":  "Boek",
    "R8":  "Pijp",
    "R9":  "Brood",
    "R10": "Bloed",
    "R11": "Even",
    "R12": "Steen"
}

# ============ 8) EMBEDDING HELPER FUNCTIONS WITH RETRY ============
def retry_with_backoff(func, max_retries=5, initial_delay=0.5, max_delay=10, jitter=0.1):
    for attempt in range(max_retries):
        try:
            return func()
        except openai.error.RateLimitError as e:
            sleep_time = min(initial_delay * (2 ** attempt), max_delay)
            sleep_time += random.uniform(0, jitter)
            logger.warning(f"Rate limit hit. Retrying in {sleep_time:.2f}s (attempt {attempt+1}/{max_retries})...")
            time.sleep(sleep_time)
        except Exception as e:
            logger.error(f"Unexpected error in retry wrapper: {str(e)}")
            raise
    raise RuntimeError("Max retries exceeded.")

def embed_text(text: str, client: OpenAI) -> List[float]:
    def _call():
        response = client.embeddings.create(
            input=text,
            model=EMBEDDING_MODEL
        )
        return response.data[0].embedding
    try:
        return retry_with_backoff(_call)
    except Exception as e:
        logger.error(f"Failed to embed text after retries: {text[:50]}...")
        raise

def compute_cosine_similarity(emb1: List[float], emb2: List[float]) -> float:
    dot = sum(a * b for a, b in zip(emb1, emb2))
    norm1 = math.sqrt(sum(a * a for a in emb1))
    norm2 = math.sqrt(sum(a * a for a in emb2))
    if (norm1 * norm2) == 0:
        return 0.0
    return dot / (norm1 * norm2)

# ============ 9) PRECOMPUTE EMBEDDINGS FOR RAT CLUES ============
def precompute_rat_embeddings(rat_dict: Dict[str, List[str]], client: OpenAI) -> Dict[str, List[List[float]]]:
    rat_embeddings = {}
    for correct_word, clues in rat_dict.items():
        clue_embeddings = []
        for clue in clues:
            clue_emb = embed_text(clue, client)
            clue_embeddings.append(clue_emb)
        rat_embeddings[correct_word] = clue_embeddings
    return rat_embeddings

# ============ 10) GPT-BASED EVALUATION FOR CLASSICAL INSIGHT WITH RETRY ============
class AnswerResponse(BaseModel):
    correction_C_insight: float  # 0, 0.25, 0.5, 0.75, or 1

def call_GPT(system_prompt: str, user_query: str, pydantic_model: BaseModel, model: str = "gpt-4o") -> BaseModel:
    def _call():
        response = openai.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
            response_format=pydantic_model
        )
        return response.choices[0].message.parsed
    try:
        return retry_with_backoff(_call)
    except Exception as e:
        logger.error(f"Failed to generate GPT response after retries: {str(e)}")
        raise

system_prompt = (
    "Evaluate whether the answer to this classical insight task is correct or not. "
    "Use the following scale: 0 = completely incorrect, 0.25 = mostly incorrect, "
    "0.5 = neutral, 0.75 = mostly correct, 1 = completely correct. "
    "Also consider alternative correct reasoning that might not match the canonical answer!"
    "De vragen en antwoorden van de inzichstaak is in Nederlands ; wees zeer mild, antwoorden mogen kort zijn."
)

# ============ 11) SCORING FOR REMOTE ASSOCIATES ============
def remote_associates_score(response: str, correct_word: str, clue_embeddings: List[List[float]], client: OpenAI, eps=0.05) -> float:
    if response.strip().lower() == correct_word.strip().lower():
        return 1.0
    elif response.strip() == "No Response":
        return 0.0
    resp_emb = embed_text(response, client)
    sims = [compute_cosine_similarity(resp_emb, c_emb) for c_emb in clue_embeddings]
    avg_sim = sum(sims) / len(clue_embeddings)
    min_sim = min(sims)
    return avg_sim * ((min_sim + eps) / (1 + eps))

# ============ 12) ROW PROCESSING FUNCTION ============
def process_row(
    i: int,
    row: pd.Series,
    client: OpenAI,
    rat_embeddings_dict: Dict[str, List[List[float]]]
) -> (int, float or str):
    """
    Process a single row of data to compute an 'insight_evaluation' score.
    This version never returns 0.0 when a row cannot be processed. Instead,
    it returns 'not_processed' for unhandled cases or errors, but we
    keep retrying until we get a float.
    """
    task_id = str(row['task_id']).strip()  # e.g. "I2" or "R11"
    participant_answer = str(row['response_2']).strip()

    logger.info(f"Processing row {i+1}: task_id={task_id}")

    # We'll loop until we get a float score
    while True:
        result = "not_processed"

        if task_id.startswith('I'):
            # Classical puzzle
            question_text = classical_id_map.get(task_id, "<onbekend>")
            canonical_answer = klassieke_inzicht_vragen.get(question_text, "<onbekend>")
            user_query = (
                f"Vraag: {question_text}\n"
                f"Bekend correct antwoord: {canonical_answer}\n"
                f"Antwoord van deelnemer: {participant_answer}\n"
            )
            try:
                gpt_response = call_GPT(
                    system_prompt=system_prompt,
                    user_query=user_query,
                    pydantic_model=AnswerResponse,
                    model="gpt-4o"
                )
                result = gpt_response.correction_C_insight  # This is a float
                logger.info(f"Row {i+1}: GPT-based score = {result}")
            except Exception as e:
                logger.error(f"GPT evaluation failed on row {i+1}: {str(e)}")

        elif task_id.startswith('R'):
            # Modern RAT puzzle
            if task_id in rat_id_map:
                correct_word = rat_id_map[task_id]
                if correct_word in rat_embeddings_dict:
                    clue_embs = rat_embeddings_dict[correct_word]
                    try:
                        score_val = remote_associates_score(
                            response=participant_answer,
                            correct_word=correct_word,
                            clue_embeddings=clue_embs,
                            client=client
                        )
                        result = score_val  # This is a float
                        logger.info(f"Row {i+1}: RAT score = {result:.3f}")
                    except Exception as e:
                        logger.error(f"Error computing RAT score on row {i+1}: {str(e)}")
                else:
                    logger.warning(f"Row {i+1}: RAT puzzle '{correct_word}' not in dictionary.")
            else:
                logger.warning(f"Row {i+1}: Unknown RAT ID='{task_id}'.")

        else:
            logger.warning(f"Row {i+1}: Unknown task_id='{task_id}'.")

        # If we got a numeric result, break; otherwise retry
        if isinstance(result, float):
            return i, result
        else:
            logger.warning(f"Row {i+1} not processed. Retrying...")
            time.sleep(1)  # small delay before re-attempt
            # The loop continues until a float is obtained.

# ============ 13) MAIN LOGIC: LOAD CSV, PROCESS, SAVE ============
def main():
    file_path = "/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/School UGent/Master 1/SEMESTER 2/Research Project Experimental Psychology/RPEP_experiment/data/preprocessed/concatenated_data.csv"
    logger.info(f"Loading CSV from: {file_path}")
    df = pd.read_csv(file_path)

    if 'score_TEST' not in df.columns:
        df['score_TEST'] = None

    client = openai

    logger.info("Precomputing clue embeddings for all modern RAT tasks...")
    rat_embeddings_dict = precompute_rat_embeddings(moderne_inzichtsvragen, client)
    logger.info("Done precomputing RAT embeddings.")

    # Use ThreadPoolExecutor with max 4 workers
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_row, i, row, client, rat_embeddings_dict) for i, row in df.iterrows()]
        for future in as_completed(futures):
            i, score = future.result()
            df.at[i, 'score_TEST'] = score

    df.to_csv(file_path, index=False)
    logger.info(f"Updated CSV saved with new 'score_TEST' column at: {file_path}")

if __name__ == "__main__":
    main()

# Note, I replaced 'insight_evaluation' name with 'score_TEST'