# 3.1 (0.5 points)

Fill in the rest of the table below:

|      | they | can | can | fish | END |
|------|------|-----|-----|------|-----|
| Start|  n/a | n/a | n/a | n/a  | n/a |
| Noun | -2   | -10 | -10 | -15  | n/a |
| Verb | -13  | -6  | -11 | -16  | n/a |
| End  | n/a  | n/a | n/a | n/a  | -17 |


# 4.2 (0.5 points)

Do you think the predicted tags "PRON AUX AUX NOUN PUNCT" for the sentence "They can can fish ." are correct? Use your understanding of parts-of-speech from the notes.  

According to me, it should be "PRON AUX VERB NOUN PUNCT", as the second "can" is verb "to can fish" and the first one explains the ability of "they" to "can fish". 



# 8 (7650 only; 1 point)

Find an example of sequence labeling for a task such as part-of-speech tagging, in a paper at ACL, NAACL, EMNLP, EACL, or TACL, within the last five years. 

### Part-of-Speech Tagging for Twitter with Adversarial Neural Networks
Tao Gui Qi Zhang Haoran Huang Minlong Peng Xuanjing Huang  
Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing   

* **What is the task they are trying to solve?**  
    The paper addresses the problem of part-of-speech tagging for Tweets.   
  
* **What tagging methods do they use? HMMs CRF? max-margin markov networks deep learning models?**   
    The paper uses a sequence-to-sequence autoencoder to perform tagging.  
  
* **What features do they use and why?**
    The paper proposes a method that tries to learn common features through adversarial discriminator similar to that in Adversarial Neural Networks. The reason is that there is a lack of large scale labeled datasets for the domain and  that tweets are usually informal and contain numerous out-of-vocabulary words  
  
* **What methods and features are most effective?** . 
    The paper claims that their proposed method achieves the state-of-the-art results with much less labelled data  
  
* **Give a one-line summary of the paper that the authors are trying to leave for the reader.**  
    The key idea about the paper is the use of adversarial networks to construct features for weakly labelled data. The authors make use of abundant out-of-domain labeled data(Newswire articles) abundant  unlabeled in-domain data(Tweets) and scarce labeled indomain data(Tweets) to perform the POS Tagging.  

#### **Paper Summary** 

The paper recognizes the sparsely limited labelled data in the domain of tweets. Additionally the number of out of vocabulary words in tweets is remarkably high as compared to other language sources like newswire articles. The number of misspelled words in tweets also poses as a challenge for tagging. The paper proposes to use out-of-domain labeled data(Newswire articles) in addition to the sparsely labelled indomain data(Tweets) and build a POS tagger. In doing so they make use of BiLSTMs adversarial neural networks and CNNs to build **Target Preserved Adversarial Neural Network**.

**Approach**
The TPANN architecture is split into 4 sections
* **Feature Extractor(F)** adopts CNN to extract character embedding features which can tackle the out-of-vocabulary word problem effectively. To incorporate word embedding features a concatenation of word embeddings and character embedding is passed as an input to the bi-LSTM on the next layer. Utilizing a bi-LSTM to model sentences the feature extractor can extract sequential relations and context information. 
    
* **POS Tagging Classifier(P)** and **Domain Discriminator(Q)** take the output of the Feature extractor as the input. They are standard feed-forward networks with a softmax layer for classification. P predicts POS tagging label to get classification capacity and Q discriminates domain label to make F(x) domain-invariant

* **Target Domain Autoencoder(R)** Through training adversarial networks domain-invariant features can be obtained but they may weaken some domain-specific features useful for POS tagging classification. Merely obtaining domain invariant features would therefore limit the classification ability. The proposed model tries to tackle this defect by introducing domain-specific autoencoder R which attempts to reconstruct target domain data  

   