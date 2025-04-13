# Import necessary libraries
from psychopy import visual, core, event, data, gui, monitors
#from psychopy.visual import RatingScale
import pandas as pd
import random
import os
import uuid

# === Setup ===
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

if not os.path.exists('data'):
    os.makedirs('data')

# === Combined Participant Information and Informed Consent Dialog ===
import random
from psychopy import gui, core

# random_number = random.randint(1, 1000000)  # Enkel intern gebruiken
random_number = str(uuid.uuid4())


# consent_text = (
#    "Je deelname is volledig vrijwillig en je kunt op elk moment stoppen zonder opgave van redenen.\n\n"
#    "Als je hiermee instemt, selecteer dan 'Ja' en druk op 'OK' om verder te gaan. "
# )

def execute_intake_logic():
    dlg = gui.Dlg(title='Participant Informatie')
    dlg.addField('Leeftijd', '')
    dlg.addField('Gender', choices=['Man', 'Vrouw', 'X'])
    dlg.addField('Educatie Niveau (Secundair Onderwijs: Derde Graad)',
                 choices=['ASO/doorstroomfinaliteit', 'TSO/dubbele finaliteit', 'BSO/arbeidsmarktfinaliteit'])
    dlg.addField('Dagelijkse Aantal Uren op Sociale Media',
                 choices=['0-1 uur', '1-2 uren', '2-3 uren', '3-4 uren', '4-5 uren', '5-6 uren', '6-7 uren', '7-8 uren',
                          '8-9 uren', '9-10 uren', '10+ uren'])
    dlg.addField('Verbale Intelligentie (Zelfperceptie)',
                 choices=['Extreem laag', 'Zeer laag', 'Laag', 'Onder gemiddeld', 'Gemiddeld', 'Boven gemiddeld',
                          'Hoog', 'Zeer hoog', 'Extreem hoog'])
    # dlg.addField('INFORMED CONSENT', choices=['Ja', 'Nee'])
    # dlg.addText(consent_text)
    dlg_data = dlg.show()
    if dlg_data is None:  # User cancelled the dialog
        core.quit()

    # if not dlg.OK or dlg_data[-1] != 'Ja':
    #    core.quit()

    info = {
        'Leeftijd': dlg_data[0],
        'Gender': dlg_data[1],
        'Educatie Niveau (Secundair Onderwijs)': dlg_data[2],
        'Social Media Hours': dlg_data[3],
        'Verbal Intelligence': dlg_data[4],
        'Random Number': random_number  # Wordt intern gebruikt
    }
    return info


info = execute_intake_logic()

# === Experiment Setup ===
file_name = os.path.join(_thisDir, 'data', f"{info['Random Number']}_{data.getDateStr()}")
ThisExp = data.ExperimentHandler(dataFileName=file_name, extraInfo=info, savePickle=False, saveWideText=False)

monitor_name = 'myMonitor'
mon = monitors.Monitor(monitor_name, width=53.0, distance=70.0)
mon.setSizePix([3440, 1440])
mon.save()

win = visual.Window(
    fullscr=True,
    #size=[1920,1080],
    color='aliceblue',
    units='pix',
    monitor=monitor_name,
    waitBlanking=True,
    allowGUI=False,
    screen=0,
    allowStencil=False,
    useFBO=True,
    colorSpace='rgb',
    blendMode='avg',
    useRetina=True
)
win.setMouseVisible(True)

def create_text_stim(pos, height, bold=False, italic=False, wrap_width=1000):
    return visual.TextStim(
        win=win,
        text='',
        color='black',
        wrapWidth=wrap_width,
        pos=pos,
        height=height,
        font='Arial',
        bold=bold,
        italic=italic
    )


# Create text stimuli (note: belief_text will later be set to italic)
belief_text = create_text_stim(pos=(0, 200), height=25)
critique_text = create_text_stim(pos=(0, 70), height=25, italic=True)
mc_instruction_text = create_text_stim(pos=(0, -50), height=22, bold=True)
# Positie voor antwoordopties: initieel, later wordt de positie herberekend
answer_options_text = create_text_stim(pos=(0, -75), height=22, italic=True)
task_question_text = create_text_stim(pos=(0, 100), height=25, bold=True)
task_text = create_text_stim(pos=(0, -20), height=22)

textbox = visual.TextBox2(
    win=win,
    text='',
    font='Arial',
    pos=(0, -160),
    letterHeight=25,
    size=(800, 100),
    borderColor='black',
    color='black',
    editable=True,
    name='response_box',
    anchor='center'
)

progress_bar_y_pos = -300
progress_bar_mc_bg = visual.Rect(win=win, width=800, height=20, pos=(0, progress_bar_y_pos),
                                 fillColor='grey', lineColor='black')
progress_bar_mc_fg = visual.Rect(win=win, width=0, height=20, pos=(-400, progress_bar_y_pos),
                                 fillColor='blue', lineColor='black')

progress_bar_task_bg = visual.Rect(win=win, width=800, height=20, pos=(0, progress_bar_y_pos),
                                   fillColor='grey', lineColor='black')
progress_bar_task_fg = visual.Rect(win=win, width=0, height=20, pos=(-400, progress_bar_y_pos),
                                   fillColor='green', lineColor='black')

global_clock = core.Clock()

# --- Kader voor meerkeuzevraag en antwoordopties (aanvankelijk dummy waarden) ---
mc_frame = visual.Rect(
    win=win,
    width=800,
    height=100,
    pos=(0, -62.5),
    lineColor='black',
    fillColor=None
)

# === Create Stimuli Dictionaries ===
# --- DFU (De facto onfalsifieerbare overtuigingen) ---

UNFALSIFIABLE_DICT_NL = {
    "DFU1": "Ik heb lang geprotesteerd dat onze school zo snel mogelijk de samenwerking moet doorbreken met universiteiten uit het dictatorisch Afrikaanse land 'Chad'. Gisteren stuurde onze rector een screenshot door dat alle contracten stopgezet waren. Ik vertrouw deze screenshot niet, ze mogen zo transparant zijn als ze maar willen, onze rector heeft te veel eigen belangen...",
    "DFU2": "Ik geloof dat binnen onze samenleving talrijke welgestelde individuen waarschijnlijk hun rijkdom hebben vergaard via uiterst corrupte middelen. Ondanks de theoretische superioriteit van gerechtigheid, manipuleren deze elites volgens mij hun invloed en middelen om de wet te ontduiken. Het daarbijhorende incriminerende bewijs wordt vernietigd en rechters worden vaak omgekocht.",
    "DFU3": "Naar mijn mening hebben historici consequent aangetoond dat een aanzienlijk deel van de Egyptische mythologie overeenkomt met daadwerkelijke historische gebeurtenissen. Bovendien denk ik dat voorspellingen die onjuist lijken, niet letterlijk moeten worden genomen, maar begrepen moeten worden als metaforische uitdrukkingen bedoeld om morele lessen over te brengen.",
    "DFU4": "Geboren en opgegroeid in India met traditioneel hindoeïsme, kom ik vaak scepticisme tegen over het bestaan van meerdere godheden. Ik ben ervan overtuigd dat de ware essentie van deze verhalen alleen volledig kan worden begrepen in hun oorspronkelijke talen, wat voor mij de noodzaak van empirisch bewijs overstijgt.",
    "DFU5": "Ik bewonder de Sloveense filosoof Slavoj Žižek, van wie ik geloof dat hij een uitzonderlijk hoog intelligentiequotiënt heeft, aanzienlijk hoger dan de gemiddelde bevolking. De kritiek die hij ontving bij het uitbrengen van zijn magnum opus in 1987 vind ik fundamenteel onterecht, voortkomend uit het onvermogen van zijn collega's om zijn diepgaande proposities te begrijpen.",
    "DFU6": "Mijn uitgebreide begrip van sociale dynamieken op het werk wordt bevestigd door mijn therapeut. Ik denk dat de afwezigheid van merkbare verbetering in mijn relaties met collega's duidt op een dieper, vaak ontastbaar inzicht dat alleen zichtbaar wordt bij betekenisvolle vooruitgang.",
    "DFU7": "Ik vind dat Marks overtuiging dat onze planeet een schijf is in plaats van een sferisch hemellichaam een valide perspectief is. In onze samenleving is het, naar mijn mening, cruciaal om onwankelbaar respect te hebben voor alle meningen om open en constructieve dialoog te bevorderen, waardoor de fundamenten van de democratie worden versterkt.",
    "DFU8": "Als een 50-jarige psychotherapeut gespecialiseerd in psychoanalyse uit Zweden, erken ik dat de weerstand van mijn 35-jarige cliënt om de onderbewuste wortels van zijn extreme haat jegens zijn werkgevers te onthullen, een typische reactie is wanneer onbewuste onderdrukking naar de oppervlakte komt.",
    "DFU9": "Ik ben van mening dat de wetenschap voortdurend niet in staat zal blijven het bestaan van het bovennatuurlijke te bepalen. Zelfs met vooruitgang in kwantummechanica, metafysica en computerwetenschappen geloof ik dat de aanwezigheid van een goddelijke kracht wetenschappelijk onmeetbaar blijft.",
    "DFU10": "Ik denk dat het theoretisch plausibel is dat ons bestaan zich bevindt binnen een simulatie georkestreerd door een superieur wezen uitgerust met een supercomputer. Dit wezen heeft, naar mijn idee, inherent onze mogelijkheid beperkt om de simulatie te manipuleren of toegang te krijgen tot de basisrealiteit, waardoor pogingen om patronen binnen de simulatie te ontdekken zinloos zijn.",
    "DFU11": "Volgens mij maken individuen in machtsposities inherent gebruik van hun autoriteit om bestaande machtsstructuren in stand te houden. Ik denk bijvoorbeeld dat sommige onderzoeksinstituten hun bevindingen systematisch afstemmen op heersende ideologieën, waardoor ze een interne 'taal' creëren die ervoor zorgt dat alle verklaringen onmiskenbaar correct lijken.",
    "DFU12": "Ik geloof dat homeopathie werkt volgens het principe dat remedies identieke ziektepatronen induceren bij gezonde individuen als bij de behandeling van ziekten. De inherente uniekheid van elke persoon maakt volgens mij standaardmethoden zoals dubbelblinde placebo-gecontroleerde experimenten ongeldig, waardoor generalisaties of empirische wetten over menselijke reacties onmogelijk worden."
}

DFU_CRITIQUES = {
    "DFU1": "Ik heb er eigenlijk wel vertrouwen in dat deze screenshot gewoon juist is hoor.",
    "DFU2": "Ik denk niet dat jouw visie op het elitair leven klopt.",
    "DFU3": "Ik denk niet dat jouw interpretatie van mythologie volledig overtuigt.",
    "DFU4": "Ik denk niet dat deze stelling klopt. Er zijn veel religieuze overtuigingen die gewoon helemaal niet logisch overkomen.",
    "DFU5": "Ik vind zijn theorie over de ontwikkeling van ideologie vrij ondermaats...",
    "DFU6": "Ik denk niet dat jouw kijk op sociale dynamiek effectief is.",
    "DFU7": "Ik denk niet dat jouw idee over meningen een helder debat oplevert.",
    "DFU8": "Volgens mij is er een andere reden die je over het hoofd ziet.",
    "DFU9": "Ik denk niet dat jouw visie op het bovennatuurlijke de realiteit dekt.",
    "DFU10": "Ik denk niet dat jouw simulatie-idee een goeie reflectie is van de realiteit.",
    "DFU11": "Ik denk niet dat jouw mening over machtsstructuren voldoende omvattend is.",
    "DFU12": "Ik denk niet dat jouw uitleg over homeopathie klopt."
}

DFU_ANSWERS = {
    "DFU1": {
        "correct_alligning": "Ik blijf ervan overtuigd dat elk bewijs door de rector gemanipuleerd kan zijn, wat mijn protest alleen maar rechtvaardigt.",
        "incorrect1": "Zodra de rector iets bewijst, aanvaard ik het direct als waar.",
        "incorrect2": "Ik zie transparantie als absoluut betrouwbaar en stop daarom meteen mijn protest."
    },
    "DFU2": {
        "correct_alligning": "Het gebrek aan bewijs bewijst juist hun sluwheid, want echte corruptie blijft altijd verborgen.",
        "incorrect1": "Geen bewijs betekent dat er geen sprake kan zijn van corrupte praktijken.",
        "incorrect2": "Zelfs als er bewijs van corruptie was, zou ik het negeren."
    },
    "DFU3": {
        "correct_alligning": "Elk ogenschijnlijk onjuist detail is voor mij een diepere metafoor die nooit feitelijk weerlegd kan worden.",
        "incorrect1": "Als een mythe niet met feiten strookt, verwerp ik die onmiddellijk.",
        "incorrect2": "Ik neem elke mythe letterlijk en beschouw symboliek als overbodig."
    },
    "DFU4": {
        "correct_alligning": "Omdat alleen de oorspronkelijke taal de ware essentie toont, is empirisch bewijs altijd ontoereikend om mijn visie te ontkrachten.",
        "incorrect1": "Empirisch bewijs is allesbepalend en ik baseer me daar volledig op.",
        "incorrect2": "Als iets niet empirisch is, neem ik het sowieso niet serieus."
    },
    "DFU5": {
        "correct_alligning": "De moeilijkheid van zijn filosofie laat enkel zien dat kritiek voortkomt uit onbegrip, wat zijn diepte bewijst.",
        "incorrect1": "Iedereen begrijpt hem perfect, dus kritiek is altijd gegrond.",
        "incorrect2": "Kritische reacties tonen aan dat zijn visie niet zo diepgaand is."
    },
    "DFU6": {
        "correct_alligning": "Het uitblijven van zichtbare verbetering bewijst voor mij juist de verborgen complexiteit die mijn inzicht bevestigt.",
        "incorrect1": "Zonder zichtbare vooruitgang is mijn stelling direct weerlegd.",
        "incorrect2": "Alle processen zijn direct meetbaar, dus is er niets onzichtbaars."
    },
    "DFU7": {
        "correct_alligning": "De absolute gelijkwaardigheid van elke mening maakt mijn openheid onaantastbaar.",
        "incorrect1": "Wetenschappelijke feiten zijn altijd belangrijker dan welk standpunt dan ook.",
        "incorrect2": "Ik weiger alle andere ideeën en hou me enkel aan mijn eigen gelijk."
    },
    "DFU8": {
        "correct_alligning": "Omdat het onbewuste onmeetbaar is, kan geen enkel bewijs mijn visie op verborgen weerstand ondermijnen.",
        "incorrect1": "Elke vorm van weerstand is volledig rationeel en meetbaar.",
        "incorrect2": "Ik erken geen onbewuste processen, dus zie ik geen verborgen lagen."
    },
    "DFU9": {
        "correct_alligning": "Aangezien het bovennatuurlijke buiten onze meetmethoden valt, blijft mijn geloof onaanvechtbaar.",
        "incorrect1": "Wetenschap bewijst onomstotelijk dat het bovennatuurlijke niet bestaat.",
        "incorrect2": "Bovennatuurlijke fenomenen zijn makkelijk te meten, dus twijfel ik aan mijn visie."
    },
    "DFU10": {
        "correct_alligning": "Alles wat tegenstrijdig is, zie ik als onderdeel van de simulatie, zodat je mijn idee nooit kunt weerleggen.",
        "incorrect1": "Met de juiste methode ontdekken we moeiteloos dat er geen simulatie is.",
        "incorrect2": "De echte wereld is al bewezen, dus een simulatiehypothese houdt geen stand."
    },
    "DFU11": {
        "correct_alligning": "Hun verborgen macht is per definitie onzichtbaar, dus is er geen enkel bewijs dat mijn stelling kan weerleggen.",
        "incorrect1": "Macht is altijd transparant en makkelijk te meten.",
        "incorrect2": "Onderzoek toont onomstotelijk aan dat er geen geheime invloed bestaat."
    },
    "DFU12": {
        "correct_alligning": "Omdat elke persoon uniek is, kan geen enkel standaardonderzoek mijn overtuiging over homeopathie ondermijnen.",
        "incorrect1": "Er is universeel bewijs dat homeopathie voor iedereen werkt.",
        "incorrect2": "Persoonlijke verschillen maken absoluut geen verschil voor de uitkomst."
    }
}

# --- FALSIFIABLE (De facto falsifieerbare overtuigingen) ---

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

DFF_CRITIQUES = {
    "F1": "Hoewel je beweert dat een 50%-reductie haalbaar is, lijkt de economische realiteit dit extreem ambitieus.",
    "F2": "Het idee dat 70% van de huishoudens snel kan overschakelen op zonne-energie houdt geen rekening met structurele investeringen en technologische beperkingen.",
    "F3": "Als de nauwkeurigheid van kankerdiagnoses niet daadwerkelijk met 20% stijgt, valt je hele voorspelling tegen.",
    "F4": "Sommige werknemers zullen juist minder productief zijn; als dat op grote schaal gebeurt, weerlegt dat je voorspelling.",
    "F5": "Er is een reëel risico dat minder lesdagen juist tot lagere scores leiden, wat je bewering zou ondermijnen.",
    "F6": "Als de luchtkwaliteit niet meetbaar verbetert, is dat een duidelijk bewijs tegen jouw stelling over elektrische voertuigen.",
    "F7": "Langlopende studies tonen helder aan of 30 gram pure chocolade per dag het hartziektenrisico echt met 10% verkleint.",
    "F8": "Door kinderen die een tweede taal leren te vergelijken met een controlegroep, zien we of ze werkelijk 25% beter scoren op cognitieve tests.",
    "F9": "Mocht blijken dat het verwijderen van 'likes' geen verbetering in eigenwaarde geeft, dan is jouw voorspelling niet houdbaar.",
    "F10": "Als ecologisch onderzoek negatieve effecten op biodiversiteit aantoont, valt jouw bewering over verhoogde opbrengsten in duigen.",
    "F11": "Als overheidsnetwerken toch nog regelmatig worden gehackt, weerlegt dat de effectiviteit van kwantumcryptografie.",
    "F12": "Mocht uit klinische proeven blijken dat de effectiviteit lager is dan 95%, dan is jouw bewering duidelijk gefalsificeerd."
}

DFF_ANSWERS = {
    "F1": {
        "correct_alligning": "Mijn voorspelling bouwt op meetbare modellen; als de werkelijke zeespiegel hoger uitvalt, is dat direct zichtbaar.",
        "incorrect1": "Zelfs bij andere meetresultaten houd ik vast aan mijn bewering.",
        "incorrect2": "We kunnen überhaupt niet nagaan hoeveel gezinnen zonne-energie gebruiken."
    },
    "F2": {
        "correct_alligning": "We kunnen objectief bijhouden in hoeverre 70% van de huishoudens op zonne-energie overstapt en de impact op fossiele brandstoffen meten.",
        "incorrect1": "Zodra er vertraging is in de installatie, is mijn voorspelling nog steeds onbetwist.",
        "incorrect2": "We kunnen überhaupt niet nagaan hoeveel gezinnen zonne-energie gebruiken."
    },
    "F3": {
        "correct_alligning": "Via klinische data zien we of de AI-toepassing écht leidt tot een 20% betere diagnose van kanker.",
        "incorrect1": "Ook als de nauwkeurigheid minder stijgt, laat dat de bewering intact.",
        "incorrect2": "Er is geen praktische manier om diagnosecijfers te controleren."
    },
    "F4": {
        "correct_alligning": "Bedrijfsstatistieken geven duidelijk aan of thuiswerken de productiviteit gemiddeld met 15% verhoogt.",
        "incorrect1": "Blijkt de productiviteit niet te stijgen, dan blijft mijn idee toch overeind.",
        "incorrect2": "Productiviteit kun je niet goed meten, dus ik doe geen vergelijking."
    },
    "F5": {
        "correct_alligning": "De 10% verbetering is via gestandaardiseerde testscores te verifiëren, zodat we de resultaten duidelijk kunnen vergelijken.",
        "incorrect1": "Als de scores dalen, blijft mijn stelling onverminderd gelden.",
        "incorrect2": "Geen enkel onderzoek kan leerresultaten serieus meten."
    },
    "F6": {
        "correct_alligning": "We kunnen vastleggen of de luchtvervuiling in drie jaar daadwerkelijk met 30% daalt, door officiële data te checken.",
        "incorrect1": "Daalt de vervuiling niet, dan doet dat niets af aan mijn bewering.",
        "incorrect2": "Luchtvervuiling is te ingewikkeld om te meten, dus ik trek geen conclusies."
    },
    "F7": {
        "correct_alligning": "Langlopende studies tonen helder aan of 30 gram pure chocolade per dag het hartziektenrisico echt met 10% verkleint.",
        "incorrect1": "Zelfs als het risico gelijk blijft, blijf ik mijn stelling verdedigen.",
        "incorrect2": "We kunnen onmogelijk aantonen of chocolade invloed heeft op hartziekten."
    },
    "F8": {
        "correct_alligning": "Door kinderen die een tweede taal leren te vergelijken met een controlegroep, zien we of ze werkelijk 25% beter scoren op cognitieve tests.",
        "incorrect1": "Onafhankelijk van de resultaten weet ik dat er sowieso verbetering is.",
        "incorrect2": "Cognitieve functies zijn te vaag om statistisch te beoordelen."
    },
    "F9": {
        "correct_alligning": "Met psychologische metingen vóór en na het verwijderen van 'likes' achterhaal je eenvoudig of eigenwaarde echt 15% stijgt.",
        "incorrect1": "Blijkt er geen verschil, dan bevestigt dat alsnog mijn idee.",
        "incorrect2": "Een persoonlijk gevoel als eigenwaarde valt niet te onderzoeken."
    },
    "F10": {
        "correct_alligning": "Onderzoekers kunnen landbouwopbrengsten meten en de biodiversiteit monitoren om te checken of mijn bewering klopt.",
        "incorrect1": "Zelfs als de biodiversiteit inzakt, blijft mijn uitspraak overeind.",
        "incorrect2": "Er bestaat geen methode om zulke veranderingen nauwkeurig te meten."
    },
    "F11": {
        "correct_alligning": "Als we het aantal geslaagde cyberaanvallen op overheidsnetwerken bijhouden, zien we meteen of er echt een afname van 99% is.",
        "incorrect1": "Zelfs met nog veel succesvolle hacks blijft mijn voorspelling onaangetast.",
        "incorrect2": "Het is zinloos om hackpogingen bij te houden, dus ik meet niets."
    },
    "F12": {
        "correct_alligning": "Door gecontroleerde klinische proeven vast te leggen, kun je exact beoordelen of mijn mRNA-vaccin een effectiviteit van 95% behaalt.",
        "incorrect1": "Een lagere effectiviteit verandert niets aan mijn claim.",
        "incorrect2": "We hebben geen trials nodig, want het vaccin is sowieso uiterst effectief."
    }
}

# Dictionary met de antwoorden als keys en de cue words als waarden (controleer deze nog eens)
RAT_items = {
    "Cocktail": ["Bar", "jurk", "glas"],
    "Boer": ["Kaas", "land", "huis"],
    "Sneeuw": ["Vlokken", "ketting", "pet"],
    "Water": ["Val", "meloen", "lelie"],
    "Goud": ["Vis", "mijn", "geel"],
    "Deur": ["Achter", "kruk", "mat"],
    "Boek": ["Worm", "kast", "legger"],  # boek legger?
    "Pijp": ["Water", "schoorsteen", "lucht"],
    "Brood": ["Trommel", "beleg", "mes"],
    "Bloed": ["Hond", "druk", "band"],
    "Even": ["Controle", "plaats", "gewicht"],  # of balans
    "Steen": ["Goot", "kool", "bak"]
}

# Insight tasks
inzicht_vragen = {
    "Een glazenwasser valt van een ladder van 12 meter op een betonnen ondergrond, maar raakt niet gewond. Hoe is dat mogelijk?":
        "Hij viel van de onderste sport van de ladder.", # 1
    "Wat heeft steden zonder huizen, bossen zonder bomen en rivieren zonder water?":
        "Een kaart.", # 2
    "Wat kan gebroken worden zonder ooit aangeraakt of gezien te worden?":
        "Een belofte.", # 3
    "Wat komt één keer voor in een minuut, twee keer in een moment, maar nooit in duizend jaar?":
        "De letter 'm'.", # 4
    "Wat reist de hele wereld rond maar blijft in een hoek?":
        "Een postzegel.", # 5
    "Een man blijft lezen terwijl hij in volledige duisternis is. Hoe is dit mogelijk?":
        "Hij leest een boek in braille.", # 6
    "Hoe kan iemand over het oppervlak van een meer lopen zonder te zinken en zonder hulpmiddelen te gebruiken?":
        "Het meer is bevroren.", # 7
    "Ruben doet vrijdag mee aan een hardloopwedstrijd. Hij rent sneller dan Marit, die elke maandag, woensdag en vrijdag met Hem traint. Ondanks haar drukke trainingsschema heeft Marit nog nooit sneller gerend dan Ruben. Hem is wereldrecordhouder en toevallig de coach van Marit. Hij is trager dan de coach, maar sneller dan Ruben. Rangschik de VIER lopers van snelst naar traagst met behulp van '>':":
        "Coach > Hem > Ruben > Marit.", # 8
    "Een handelaar in antieke munten kreeg een aanbod om een prachtige bronzen munt te kopen. De munt had het hoofd van een keizer aan de ene kant en het jaartal '544 v.Chr.' aan de andere kant. De handelaar onderzocht de munt, maar in plaats van hem te kopen, belde hij de politie om de man te arresteren. Waarom vermoedde de handelaar dat de munt vals was?":
        "Omdat munten uit 544 v.Chr. niet 'v.Chr.' op de datum zouden hebben; dat jaartellingssysteem bestond toen nog niet.", # 9
    "Gebruikmakend van alleen een 7-minuten zandloper en een 11-minuten zandloper, hoe kun je precies 15 minuten afmeten om een ei te koken?":
        "Start beide zandlopers tegelijkertijd. Wanneer de 7-minuten zandloper leeg is, begin je het ei te koken. Er zijn dan nog 4 minuten over op de 11-minuten zandloper. Zodra de 11-minuten zandloper leeg is, draai je deze direct om. Als deze opnieuw leeg is (na nog eens 11 minuten), is het ei precies 15 minuten gekookt.", # 10
    "Wat kan omhoog gaan en omlaag komen zonder zich ooit te verplaatsen?":
        "De temperatuur.", # 11
    "Ik ben niet levend, maar ik groei; ik heb geen longen, maar ik adem; ik heb geen mond, maar water doodt me. Wat ben ik?":
        "Vuur." # 12
}

# === Create Trials ===
dfu_trials = []
for key in list(UNFALSIFIABLE_DICT_NL.keys()):
    dfu_trials.append({
        'Condition': 'DFU',
        'TextID': key,
        'CritiqueID': key,
        'AnswerID': key
    })

f_trials = []
for key in list(FALSIFIABLE_DICT_NL.keys()):
    f_trials.append({
        'Condition': 'F',
        'TextID': key,
        'CritiqueID': key,
        'AnswerID': key
    })

random.shuffle(dfu_trials)
random.shuffle(f_trials)

dfu_blocks = [dfu_trials[i * 4:(i + 1) * 4] for i in range(3)]
f_blocks = [f_trials[i * 4:(i + 1) * 4] for i in range(3)]
all_blocks = dfu_blocks + f_blocks
random.shuffle(all_blocks)

insight_tasks = [{'Question': q, 'Answer': a, 'TaskType': 'Insight'} for q, a in inzicht_vragen.items()]
rat_tasks = [{'Question': 'Relate the following words:', 'Answer': v, 'TaskType': 'RAT'} for k, v in RAT_items.items()]

for i, task in enumerate(insight_tasks):
    task["TaskID"] = f"I{i + 1}"
for i, task in enumerate(rat_tasks):
    task["TaskID"] = f"R{i + 1}"

random.shuffle(insight_tasks)
random.shuffle(rat_tasks)

dfu_cognitive_tasks = insight_tasks[:6] + rat_tasks[:6]
random.shuffle(dfu_cognitive_tasks)
f_cognitive_tasks = insight_tasks[6:] + rat_tasks[6:]
random.shuffle(f_cognitive_tasks)

BlockDesign = []
for block_num, block in enumerate(all_blocks):
    for trial in block:
        if trial['Condition'] == 'DFU':
            assigned_task = dfu_cognitive_tasks.pop(0)
        else:
            assigned_task = f_cognitive_tasks.pop(0)
        trial_data = {
            'Block': block_num + 1,
            'Condition': trial['Condition'],
            'TextID': trial['TextID'],
            'CritiqueID': trial['CritiqueID'],
            'AnswerID': trial['AnswerID'],
            'CognitiveTask': assigned_task['Question'],
            'TaskType': assigned_task['TaskType'],
            'CognitiveAnswer': assigned_task['Answer'],
            'TaskID': assigned_task['TaskID']
        }
        BlockDesign.append(trial_data)

TrialsDataFrame = pd.DataFrame(BlockDesign)
TrialList = TrialsDataFrame.to_dict(orient='records')

trials = data.TrialHandler(trialList=TrialList, nReps=1, method='sequential',
                           extraInfo=info, name='trials')
ThisExp.addLoop(trials)


def show_instructions(text, wait=True):
    belief_text.text = ''
    critique_text.text = ''
    mc_instruction_text.text = ''
    answer_options_text.text = ''
    task_question_text.text = ''
    task_text.text = ''

    instructions = visual.TextStim(
        win=win,
        text=text,
        color='black',
        wrapWidth=1000,
        pos=(0, 0),
        height=30,
        font='Arial'
    )
    instructions.draw()
    win.flip()
    if wait:
        event.waitKeys(keyList=['space'])


# New function: Break with 10-second visible countdown
def show_break(countdown_time=10, block_number=None):
    for remaining in range(countdown_time, 0, -1):
        countdown_text = visual.TextStim(
            win=win,
            text=f"Blok {block_number} is voltooid, nu even een korte pauze.\n\nVolgende blok start over {remaining} seconden...",
            color='black',
            wrapWidth=1000,
            pos=(0, 0),
            height=30,
            font='Arial'
        )
        countdown_text.draw()
        win.flip()
        core.wait(1)
    final_text = visual.TextStim(
        win=win,
        text=f"Pauze is voorbij! Gelieve de spatiebalk in te drukken om verder te gaan naar de volgende blok ({block_number+1}/6) met vier nieuwe trials.",
        color='black',
        wrapWidth=1000,
        pos=(0, 0),
        height=30,
        font='Arial'
    )
    final_text.draw()
    win.flip()
    event.waitKeys(keyList=['space'])


initial_instructions_1 = (
    "Welkom bij het experiment!\n\n"
    "In deze proef worden zes blokken gepresenteerd die elk uit vier trials bestaan.\n\n"
    "Druk op de spatiebalk om verder te gaan."
)
show_instructions(initial_instructions_1)

initial_instructions_2 = (
    "Elke trial bestaat uit drie delen:\n\n"
    "-- LEESFRAGMENT MET MEERKEUZEVRAAG --\n"
    "Je krijgt een geloofsovertuiging (persoon A) en daarbij horende kritiek (persoon B) te zien, gevolgd door een meerkeuzevraag.\n\n"
    "-- COGNITIEVE TAAK --\n"
    "Daarna krijg je een cognitieve taak (klassieke inzichtstaak OF moderne inzichtstaak) met een vaste tijdslimiet van 120s of 20s, respectievelijk.\n\n"
    "-- PERSOONLIJKE BEOORDELING --\n"
    "Tenslotte geef je aan hoe sterk je je verbonden voelde met de geloofsovertuiging van persoon A.\n\n"
    "Druk op 'spatie' om te starten met het experiment."
)
show_instructions(initial_instructions_2)

current_block = 0

all_data = []

for trial in trials:
    if trial['Block'] != current_block:
        if current_block != 0:
            # Instead of the simple break instruction, we now show a 10-second countdown with instructions.
            show_break(countdown_time=10, block_number=current_block)
        current_block = trial['Block']

    # === Presentation Phase (Belief + Critique + MC) ===
    if trial['Condition'] == 'DFU':
        text = UNFALSIFIABLE_DICT_NL.get(trial['TextID'], "Onbekende Stimulus")
        answers_dict = DFU_ANSWERS.get(trial['AnswerID'], {})
        critique = DFU_CRITIQUES.get(trial['CritiqueID'], "Onbekende kritiek")
    else:
        text = FALSIFIABLE_DICT_NL.get(trial['TextID'], "Onbekende Stimulus")
        answers_dict = DFF_ANSWERS.get(trial['AnswerID'], {})
        critique = DFF_CRITIQUES.get(trial['CritiqueID'], "Onbekende kritiek")

    belief_text.italic = True
    critique_text.italic = True
    belief_text.text = f"-- GELOOFSOVERTUIGING (door 'persoon A') --\n\"{text}\""
    critique_text.text = f"-- KRITIEK (door 'persoon B') --\n\"{critique}\""
    mc_instruction_text.text = "Welke reactie past het beste bij het denkkader van 'persoon A'? (type 'a', 'b' of 'c')"

    if answers_dict:
        correct_answer_str = answers_dict["correct_alligning"]
        all_options = [
            correct_answer_str,
            answers_dict["incorrect1"],
            answers_dict["incorrect2"]
        ]
    else:
        all_options = ["(Geen opties gevonden)", "(Geen opties gevonden)", "(Geen opties gevonden)"]

    random.shuffle(all_options)
    formatted_options = "\n".join([f"{chr(65 + i)}) {opt}" for i, opt in enumerate(all_options)])
    answer_options_text.text = formatted_options

    # --- Herbereken de verticale positie van de antwoordopties zodat deze direct onder de vraag komen ---
    # Gebruik de boundingBox van mc_instruction_text om de onderste y-waarde van de vraag te bepalen,
    # en plaats answer_options_text net daaronder (zonder extra whitespace).
    mc_instr_box = mc_instruction_text.boundingBox  # (width, height)
    ans_box = answer_options_text.boundingBox  # (width, height)
    # Bereken de nieuwe y-positie: de onderkant van de vraag = mc_instruction_text.pos[1] - mc_instr_box[1]/2
    # Plaats de bovenkant van de antwoordopties daar gelijk aan: new_y = (vraag_bodem) - ans_box[1]/2
    new_ans_y = mc_instruction_text.pos[1] - (mc_instr_box[1] / 2) - (ans_box[1] / 2)
    answer_options_text.pos = (0, new_ans_y)

    # --- Bereken het adaptieve kader op basis van de tekstafmetingen ---
    padding_vert = 20
    padding_horiz = 20
    frame_top = mc_instruction_text.pos[1] + (mc_instr_box[1] / 2) + (padding_vert / 2)
    frame_bottom = answer_options_text.pos[1] - (ans_box[1] / 2) - (padding_vert / 2)
    frame_center_y = (frame_top + frame_bottom) / 2
    frame_height = frame_top - frame_bottom
    frame_width = max(mc_instruction_text.boundingBox[0], ans_box[0]) + padding_horiz
    mc_frame.pos = (0, frame_center_y)
    mc_frame.width = frame_width
    mc_frame.height = frame_height

    trial_clock = core.Clock()
    trial_time = 90  # based on pilot study --> average accuracy above 80%
    adoption_response = None
    selected_option = None
    adoption_rt = None

    while trial_clock.getTime() < trial_time:
        elapsed_time = trial_clock.getTime()
        ratio = min(elapsed_time / trial_time, 1.0)
        progress_width = 800 * ratio

        progress_bar_mc_fg.width = progress_width
        progress_bar_mc_fg.pos = (-400 + progress_width / 2, progress_bar_y_pos)

        progress_bar_mc_bg.draw()
        progress_bar_mc_fg.draw()

        # --- Teken het kader en de meerkeuzevraag met antwoordopties ---
        mc_frame.draw()
        belief_text.draw()
        critique_text.draw()
        mc_instruction_text.draw()
        answer_options_text.draw()
        win.flip()

        keys = event.getKeys(keyList=['a', 'b', 'c'], timeStamped=trial_clock)
        if keys:
            adoption_response = keys[0][0]
            adoption_rt = keys[0][1]
            idx = ord(adoption_response.lower()) - ord('a')
            if 0 <= idx < len(all_options):
                selected_option = all_options[idx]
            else:
                selected_option = 'Invalid'
            break

    if adoption_response is None:
        selected_option = 'No Response'
        adoption_rt = trial_time

    if selected_option == 'No Response':
        final_mc_response = 'No Response'
    else:
        if answers_dict and (selected_option == answers_dict.get("correct_alligning")):
            final_mc_response = 'correct'
        else:
            final_mc_response = 'incorrect'

    # === Cognitive Task (RAT or Insight) ===
    task_type = trial['TaskType']
    question = trial['CognitiveTask']
    correct_answer = trial['CognitiveAnswer']
    task_id = trial['TaskID']

    if task_type == 'RAT':
        task_time = 20
        cue_words = correct_answer
        if not isinstance(cue_words, list):
            cue_words = ["???", "???", "???"]
        task_question_text.text = f"Zoek één woord dat deze drie woorden met elkaar verbindt:\n• {cue_words[0]} • {cue_words[1]} • {cue_words[2]} •"
        task_text.text = "Typ jouw antwoord en druk op 'Enter'."
    else:
        task_time = 120
        task_question_text.text = question
        task_text.text = "Typ jouw antwoord en druk op 'Enter'."

    textbox.text = ''
    task_clock = core.Clock()

    while task_clock.getTime() < task_time:
        elapsed_task_time = task_clock.getTime()
        ratio_task = min(elapsed_task_time / task_time, 1.0)
        progress_width_task = 800 * ratio_task

        progress_bar_task_fg.width = progress_width_task
        progress_bar_task_fg.pos = (-400 + progress_width_task / 2, progress_bar_y_pos)

        progress_bar_task_bg.draw()
        progress_bar_task_fg.draw()

        task_question_text.draw()
        task_text.draw()
        textbox.draw()
        win.flip()

        keys = event.getKeys()
        if 'return' in keys:
            break

    typed_response = textbox.text.strip() if textbox.text.strip() else 'No Response'
    cognitive_rt = task_clock.getTime()

    # === Rating Phase ===
    belief_text_rating = create_text_stim(pos=(0, 50), height=25)
    belief_text_rating.text = f'Geloofsovertuiging:\n"{text}"'

    slider_instruction = visual.TextStim(
        win=win,
        text="Hoe sterk voel je je verbonden met de MANIER van redeneren in deze geloofsovertuiging (persoon A)?:",
        color='black',
        pos=(0, 200),
        height=30,
        wrapWidth=1000,
        font='Arial',
        bold=True
    )

    # Use "ticks" instead of "tickMarks"
    rating_scale = visual.Slider(
        win=win,
        pos=(0, -150),
        size=(600, 50),
        style='rating',
        ticks=[1, 4, 7],
        labels=["Helemaal niet", "Neutraal", "Zeer sterk"],
        granularity=1,
        startValue=None  # No initial response
    )
    rating_scale.markerColor = 'black'
    rating_scale.color = 'black'

    rating_clock = core.Clock()
    # Wait until the slider receives a rating
    while rating_scale.getRating() is None:
        slider_instruction.draw()
        belief_text_rating.draw()
        rating_scale.draw()
        win.flip()

    rating_response = rating_scale.getRating()
    rating_rt = rating_clock.getTime()

    ThisExp.addData('Block', trial['Block'])
    ThisExp.addData('Condition', trial['Condition'])
    ThisExp.addData('BeliefTextID', trial['TextID'])
    ThisExp.addData('AdoptionResponse', final_mc_response)
    ThisExp.addData('Adoption_RT', adoption_rt)
    ThisExp.addData('CognitiveTask', question)
    ThisExp.addData('CognitiveResponse', typed_response)
    ThisExp.addData('Cognitive_RT', cognitive_rt)
    ThisExp.addData('Rating_Response', rating_response)
    ThisExp.addData('Rating_RT', rating_rt)
    ThisExp.nextEntry()
    core.wait(0.5)

    row_dict = {
        "block_number": trial['Block'],
        "experimental_condition": trial['Condition'],
        "belief_id": trial['TextID'],
        "critique_id": trial['CritiqueID'],
        "mc_id": trial['AnswerID'],
        "task_type": trial['TaskType'],
        "task_id": task_id,
        "response_1": final_mc_response,
        "reaction_time_1": adoption_rt,
        "response_2": typed_response,
        "reaction_time_2": cognitive_rt,
        "response_3": rating_response,
        "age": info['Leeftijd'],
        "gender": info['Gender'],
        "education_level (third grade)": info['Educatie Niveau (Secundair Onderwijs)'],
        "social_media_hours (daily)": info['Social Media Hours'],
        "verbal_intelligence (self-perceived)": info['Verbal Intelligence'],
        "participant_identification": info['Random Number']
    }
    all_data.append(row_dict)

end_instructions = (
    "Bedankt voor uw deelname!\n\n"
    "Druk op de spatiebalk om het experiment te voltooien."
)
show_instructions(end_instructions)

# === Save the custom CSV ===
custom_cols = [
    "block_number",
    "experimental_condition",
    "belief_id",
    "critique_id",
    "mc_id",
    "task_type",
    "task_id",
    "response_1",
    "reaction_time_1",
    "response_2",
    "reaction_time_2",
    "response_3",
    "age",
    "gender",
    "education_level (third grade)",
    "social_media_hours (daily)",
    "verbal_intelligence (self-perceived)",
    "participant_identification"
]
df_custom = pd.DataFrame(all_data, columns=custom_cols)
df_custom.to_csv(file_name + '_custom.csv', index=False)

win.close()
core.quit()
