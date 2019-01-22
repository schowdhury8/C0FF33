from nltk.tokenize import word_tokenize
import codecs
stop_words = ["i", "me", "my","also", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about",  "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "under", "again", "further", "then", "once", "here", "there", "when", "where", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such","only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
def clean(file_name,output_name,label,text):
    input_file = open(file_name,'r')
    output_file = open(output_name,'w')

    for line in input_file:
        x = line.split('---')
        tokens = word_tokenize(x[text])
        words = [word.lower() for word in tokens if word.isalpha()]
        words = [word for word in words if word not in stop_words]
        x[text] = ' '.join(words)
        if 'no response' != x[text] and '' != x[text]:
            output_file.write(str(','.join(x[label:])+'\n'))

clean('isear_data','isear_processed.csv',1,2)

input_file_n = open("negative.txt", encoding='ISO-8859-1')
input_file_p = open('positive.txt',encoding='ISO-8859-1')
output_file = open('labelled_pos_neg.txt','w')

neg = input_file_n.read()
pos = input_file_p.read()

neg = neg.split('\n')
pos = pos.split('\n')

for line in neg:
    output_file.write('0---' + line + '\n')
for line in pos:
    output_file.write('1---' + line + '\n')

output_file.close()

clean('labelled_pos_neg.txt','pos_neg_processed',0,1)
