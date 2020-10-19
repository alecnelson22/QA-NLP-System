# Load sentences from .txt into lists
def load_story(fname):
    read_text = False
    sentences = []
    sentence = ''
    with open(fname, 'r') as fp:
        for line in fp:
            if read_text:
                # Check if current line contains end of sentence
                if '.\n' in line:
                    sentence += line.split('.\n')[0]
                    sentences.append(sentence)
                    sentence = ''
                # Assumes period is followed by single spaces
                # TODO: This would count an abbreviation as a sentence end
                # TODO: 'data/1999-W02-5.story' line 40 needs to be handled
                elif '. ' in line:
                    s = line.split('. ')
                    for i in range(len(s)):
                        sentence += s[i]
                        if i < len(s) - 1:
                            sentences.append(sentence)
                            sentence = ''
                else:
                    sentence += line.strip('\n') + ' '
            elif 'HEADLINE' in line:
                d = line.split(':')
                headline = d[1].strip('\n')
            elif 'DATE' in line:
                d = line.split(':')
                date = d[1].strip('\n')
            elif 'STORYID' in line:
                d = line.split(':')
                id = d[1].strip('\n')
            elif 'TEXT' in line:
                read_text = True
    return {id: {'HEADLINE': headline, 'DATE': date, 'TEXT': sentences}}

s = load_story('data/1999-W02-5.story')
print('yer')
