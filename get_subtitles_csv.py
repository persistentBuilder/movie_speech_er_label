import re
import pandas as pd
import argparse
import codecs


def remove_special_character(value):
    value = value.strip()
    value = re.sub('[!#?\n]', '', value)
    #value = value.title()

    value = value.replace('<i>', '')
    value = value.replace('</i>', '')
    value = value.replace('-', '')
    value = value.replace('...', '')
    return value


def main():
    sub_path = args.subtitles_path
    with open(sub_path, 'r', errors='replace') as h:
        sub = h.readlines()

    re_pattern = r'[0-9]{2}:[0-9]{2}:[0-9]{2},[0-9]{3} -->'
    regex = re.compile(re_pattern)
    # Get start times
    start_times = list(filter(regex.search, sub))
    start_times = [time.split(' ')[0] for time in start_times]
    # Get lines
    lines = [[]]
    for sentence in sub:
        if re.match(re_pattern, sentence):
            lines[-1].pop()
            lines.append([])
        else:
            lines[-1].append(sentence)
    lines = lines[1:]         
    print(len(lines))
    complete_lines = []
    for line in lines:
        complete_line = ""
        for part in line:
            complete_line += " "+part
        complete_lines.append(remove_special_character(complete_line))
    movie = sub_path.split("/")[-1].split(".")[0]
    index = range(1,len(complete_lines)+1)    
    df = pd.DataFrame(complete_lines, columns=['sentence'])
    df.to_csv("movie_subtitles_csv/" + movie + "_subtitles.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--subtitles-path", type=str, default="")
    args = parser.parse_args()
    main()
