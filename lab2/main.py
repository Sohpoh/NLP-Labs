from charlm import train_char_lm,print_probs

mylm = train_char_lm('subtitles.txt',4)
print_probs(mylm,'atio')