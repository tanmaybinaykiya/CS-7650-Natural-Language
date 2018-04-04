## Research question

* **What is the task that is being solved?**  
    Factuality Prediction: Assessment of commitment towards a predicate in a sentence

* **Briefly (one sentence) explain the metric for success on this task.**  
    The authors use the following metrics to evaluate the performance of their methods
    * Mean Absolute Error  
    * Pearson correlation coefficient between automatic predictions and gold labels  

* **Why are dependency features expected to help with this task?**
    The ability of Dependeny Features to deal with languages that are morphologically rich and have a relatively free word order, makes it a good model to try on this task.

* **How are dependency features incorporated into the solution?**  
    The factuality prediction model is composed of the following components
    1. Augmentation of the TruthTeller lexicon with about 800 adjectival, nominal and verbal predicates, 
    2. **Syntactic re-ordering with PropS (Dependency Trees)**
    3. Application of TruthTeller on top of PropS trees

* **Does the paper evaluate whether dependency features improve performance on the downstream task? If so, what is their impact? If not, why not?**
    The paper employs dependency trees and compares with state of art methods which are based on dependency trees. They do not evaluate the performance impact of dependency trees.
