from nltk.tokenize import TreebankWordTokenizer
from collections import Counter
import nltk
from collections import OrderedDict
import copy


nltk.download('stopwords', quiet=True)


sentence = "The faster Harry got to the store, the faster Harry, the faster, would get home."
tokenizer = TreebankWordTokenizer()
tokens = tokenizer.tokenize(sentence.lower())
print(tokens)

bag_of_words = Counter(tokens)
print(bag_of_words)
print('Most common words:', bag_of_words.most_common(4))

times_harry_appears = bag_of_words['harry']
num_unique_words = len(bag_of_words)
tf = times_harry_appears / num_unique_words
# tf ~ term frequency
print('TF("harry") =', round(tf, 4))

kite_text = "A kite is traditionally a tethered heavier-than-air craft with wing surfaces that react" \
            "against the air to create lift and drag." \
            " A kite consists of wings, tethers, and anchors."\
            "Kites often have a bridle to guide the face of the kite at the correct angle so the wind can lift it." \
            " A kite’s wing also may be so designed so a bridle is not needed; when kiting a sailplane for launch, " \
            "the tether meets the wing at a single point. A kite may have fixed or moving anchors. Untraditionally " \
            "in technical kiting, a kite consists of tether-set-coupled wing sets; even in technical kiting, though, " \
            "a wing in the system is still often called the kite. " \
            "The lift that sustains the kite in flight is generated when air flows around the kite’s surface, " \
            "producing low pressure above and high pressure below the wings. The interaction with the wind also " \
            "generates horizontal drag along the direction of the wind. The resultant force vector from the lift and " \
            "drag force components is opposed by the tension of one or more of the lines or tethers to which the kite " \
            "is attached. The anchor point of the kite line may be static or moving (such as the towing of a kite by " \
            "a running person, boat, free-falling anchors as in paragliders and fugitive parakites or vehicle). " \
            "The same principles of fluid flow apply in liquids and kites are also used under water. " \
            "A hybrid tethered craft comprising both a lighter-than-air balloon as well as a kite " \
            "lifting surface is called a kytoon. Kites have a long and varied history and many different types " \
            "are flown individually and at festivals worldwide. Kites may be flown for recreation, art or other " \
            "practical uses. Sport kites can be flown in aerial ballet, sometimes as part of a competition. Power " \
            "kites are multi-line steerable kites designed to generate large forces which can be used to power " \
            "activities such as kite surfing, kite landboarding, kite fishing, kite buggying and a new trend snow " \
            "kiting. Even Man-lifting kites have been made."

tokens = tokenizer.tokenize(kite_text.lower())
tokens_count = Counter(tokens)
print('\n\n\n>>> Kite text tokens <<<')
print(tokens_count)

stopwords = nltk.corpus.stopwords.words('english')
tokens = [x for x in tokens if x not in stopwords]
kite_count = Counter(tokens)
print('>>> Kite text after removing stopwords')
print(kite_count)


print("\n\n\n>>> Vectorizing")
document_vector = []
doc_length = len(tokens)
for key, value in kite_count.most_common():
    document_vector.append(value / doc_length)
print("Document vector:", document_vector)


docs = list()
docs.append("The faster Harry got to the store, the faster and faster Harry would get home.")
docs.append("Harry is hairy and faster than Jill.")
docs.append("Jill is not as hairy as Harry.")

doc_tokens = []
for doc in docs:
    doc_tokens += [sorted(tokenizer.tokenize(doc.lower()))]
print("Len doc_tokens[0]:", len(doc_tokens[0]))

all_doc_tokens = sum(doc_tokens, [])
print("Len all_doc_tokens:", len(all_doc_tokens))

lexicon = sorted((set(all_doc_tokens)))
print("Len lexicon:", len(lexicon))
print(lexicon)

zero_vector = OrderedDict((token, 0) for token in lexicon)
print("Zero vector:", zero_vector)

doc_vectors = []
for doc in docs:
    vec = copy.copy(zero_vector)
    tokens = tokenizer.tokenize(doc.lower())
    token_counts = Counter(tokens)
    for key, value in token_counts.items():
        vec[key] = value / len(lexicon)
    doc_vectors.append(vec)


