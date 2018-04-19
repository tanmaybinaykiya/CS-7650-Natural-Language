# Answer to Deliverable 4.6:

## What form of semantics they are trying to capture (e.g., synonymy, hypernymy, predicate-argument, distributional)
### Distributional

## How they formalize semantics into features, constraints, or some other preference?
## They represent the semantics in terms of mentions, entitities and types.
  * **Mentions:** A mention is an observed textual reference to a latent real-world entity. Mentions are associated with nodes in a parse tree and are typically realized as NPs
    * Proper Mentions (NAM)
    * Nominal Mentions (NOM
    * Prononominal Mentions (PRO)
    Mentions are represented as key-value pairs <(r, w<sub>r</sub>)>, where keys (r) are properties while values (w<sub>r</sub>) are words. The set of properties is denoted as \R
for example, <(NAM-HEAD, Obama), (NN-MOD, Mr.)>  or <(NOM-HEAD, president)>
  * **Entities:** An entity is a specific individual or object in the world
    Entities are mappings from properties r \in \R to lists of "canonical" words used for the entity. eg: < (NAM-HEAD: [Obama, Barack], (NOM-HEAD: [president, leader])) >
* **Types:**  The class of the entity. Eg PERSON, ORGANIZATION, etc
  Types allow the sharing of properties across entities and mediate the generation of entities in our model
  Types are represented as a mapping between properties r and pairs of multinomials ![(\theta_r, f_r)](https://latex.codecogs.com/gif.latex?(\theta_r,\f_r)) where \theta_r is a unigram distribution of words that are semantically licensed for property r, f_r is a fertility distribution over the integers

### How much it helps?
  The model is evaluated on using standard coreference data sets derived from the ACE corpora.
  The paper compares results with other models and beats the other models in the precision, recall and F1 scores of all other SOTAs by a about 3%.
