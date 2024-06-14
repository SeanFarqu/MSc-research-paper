# MSc-research-paper

# Overdispersion in Count Modelling with an NHL Draft Application: Measuring Player Success

## Author
Sean Farquharson (sfarqu2@uwo.ca)  
Supervised by Dr. Douglas Woolford  

## Institution
Department of Statistical and Actuarial Sciences  
University of Western Ontario  
London, Ontario, Canada  

## Date
April 2021

## Abstract
This project investigates the use of statistical learning to predict the success of NHL players based on historical draft data. Due to the overdispersion and zero-inflation present in the data, traditional count models are not suitable. The project explores various statistical methods, including zero-inflated and hurdle models, to address these issues and provide better insights into player success.

## Contents
1. Introduction
2. Problem and Objective
3. Data
4. Methods
   - Software
   - Statistical Methods
5. Results
6. Discussion and Conclusions
7. Internship Experience at Statistics Canada

## Methods
- **Count Modelling**: Initial exploration using Poisson and Negative Binomial models.
- **Zero-Inflated Modelling**: Addressing overdispersion and excess zeros with Zero-Inflated Poisson and Zero-Inflated Negative Binomial models.
- **Hurdle Modelling**: Utilizing two-part models to better handle zero-inflated data, specifically using Truncated Poisson and Truncated Negative Binomial distributions.

## Results
- The Poisson and Negative Binomial models were found to be inadequate due to overdispersion.
- Zero-Inflated models provided a better fit but still faced challenges.
- Hurdle models, particularly with a zero-truncated negative binomial distribution, proved to be the most effective for this data.

## Conclusion
The study concludes that hurdle models are best suited for zero-inflated count data in the context of predicting NHL player success. This approach provides a more accurate and reliable framework for modeling such data.

## Acknowledgements
This project was completed under the supervision of Dr. Douglas Woolford and supported by the Department of Statistical and Actuarial Sciences at the University of Western Ontario.

## References
For a detailed list of references, please refer to the full report.
