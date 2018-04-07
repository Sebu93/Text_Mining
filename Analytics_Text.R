
######Reading Train Data
setwd("C:/Users/sebastian.pathrose/Desktop/Resume/My/Sentiment Analysis")
text_data = read.csv("train_E6oV3lV.csv",stringsAsFactors = FALSE)
text_data = text_data[,-1]

##Removing Specil smiley chara from train data
text_data$tweet <- iconv(text_data$tweet, 'UTF-8', 'ASCII')
text_data = na.omit(text_data)
View(text_data)

library(qdap)
library(tm)

##Exploratory text mining
freq_terms = freq_terms(text_data$tweet,20)
plot(freq_terms)

text_data$tweet = tolower(text_data$tweet)
text_data$tweet = removePunctuation(text_data$tweet)
text_data$tweet = removeNumbers(text_data$tweet)
text_data$tweet = stripWhitespace(text_data$tweet)
text_data$tweet = replace_abbreviation(text_data$tweet)
text_data$tweet = replace_contraction(text_data$tweet)



text_data$tweet = removeWords(text_data$tweet,stopwords("en"))
text_data$tweet = removeWords(text_data$tweet,"u")



text_data$label = as.factor(text_data$label)
table(text_data$label)
str(text_data)

#Creating source file

test_source = VectorSource(text_data$tweet)
test_source

#Creating corpse
test_corpus = VCorpus(test_source)

test_corpus[[1]][1]




# Creating DTM mATRIX

test_tdm_c = DocumentTermMatrix(test_corpus)

#Sparsing
sparse_DTM <- removeSparseTerms(test_tdm_c, 0.995)
sparse_DTM
tweetsSparse <- as.data.frame(as.matrix(sparse_DTM))
colnames(tweetsSparse) = make.names(colnames(tweetsSparse))
tweetsSparse$label = text_data$label



#######Machine model

#Train test split
library(caret)
train_test_split = createDataPartition(tweetsSparse$label,times=1,p=0.7,list = FALSE)
train_tweet = tweetsSparse[train_test_split,]
test_tweet = tweetsSparse[-train_test_split,]

glm_model_new = glm(label~ ., data=train_tweet,family = binomial)
p = predict(glm_model,test_tweet,method="class")
p_new = ifelse(p>0.5,1,0)
table(p_new)
p_new = as.data.frame(p_new)
View(p_new)

#Reading Test data for prediction

test_tweet_origin = read.csv("test_tweets_anuFYb8.csv")
library(dplyr)


test_tweet_origin = select(test_tweet_origin,tweet)
View(test_tweet_origin)
test_tweet_origin$tweet <- iconv(test_tweet_origin$tweet, 'UTF-8', 'ASCII')
test_tweet_origin = na.omit(test_tweet_origin)

sum(is.na(test_tweet_origin))
test_source = VectorSource(test_tweet_origin)
test_corpus_new = VCorpus(test_tweet_origin)

test_tweet_origin$tweet = tolower(test_tweet_origin$tweet)
test_tweet_origin$tweet = removePunctuation(test_tweet_origin$tweet)
test_tweet_origin$tweet = removeNumbers(test_tweet_origin$tweet)
test_tweet_origin$tweet = stripWhitespace(test_tweet_origin$tweet)
test_tweet_origin$tweet = replace_abbreviation(test_tweet_origin$tweet)
test_tweet_origin$tweet = replace_contraction(test_tweet_origin$tweet)



test_tweet_origin$tweet = removeWords(test_tweet_origin$tweet,stopwords("en"))
test_tweet_origin$tweet = removeWords(test_tweet_origin$tweet,"u")



text_data$label = as.factor(text_data$label)


test_source = VectorSource(test_tweet_origin$tweet)
test_corpus = VCorpus(test_source)


test_corpus = tm_map(test_corpus, tolower)
test_corpus[[1]][1]

test_corpus = tm_map(test_corpus, PlainTextDocument)
test_corpus[[1]][1]

test_corpus <- tm_map(test_corpus, stemDocument)

test_tdm_c = DocumentTermMatrix(test_corpus)
sparse_DTM <- removeSparseTerms(test_tdm_c, 0.995)

tweetsSparse <- as.data.frame(as.matrix(sparse_DTM))
colnames(tweetsSparse) = make.names(colnames(tweetsSparse))

table(tweetsSparse)


p_new_1 = predict(glm_model_new,tweetsSparse,method ="class")


p_new_1 = ifelse(p_new_1>0.5,1,0)
table(p_new_1)
