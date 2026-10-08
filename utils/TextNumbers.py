import re

class TextNumbers:
    UNITS = {
        "zero": 0, "one": 1, "two": 2, "double":2, "doubles":2, "twice":2,
        "three": 3, "triples":3, "triple":3, "thrice":3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9
    }
    TEENS = {
        "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19
    }
    TENS = {
        "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
        "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90
    }
    FRACTIONS = {
        "half": "1/2", "one-half": "1/2", "one third": "1/3", "one-third": "1/3",
        "two thirds": "2/3", "two-thirds": "2/3", "three-fourths": "3/4",
        "three fourths": "3/4", "three-quarters": "3/4", "one-fourths": "1/4",
        "one fourths": "1/4", "one-quarter": "1/4",
    }
    ALL_NUMBER_WORDS = sorted( # Combine all for regex pattern:
        list(UNITS) + list(TEENS) + list(TENS) + list(FRACTIONS),
        key=lambda x: -len(x)
    )
    COMPOUND_NUMBER_PATTERN = r'\b(?:' + '|'.join(TENS) + r')[ -](?:' + '|'.join(UNITS) + r')\b' # Add compound (e.g. "thirty one", "twenty five", with hyphens or spaces)
    FRACTIONS_PATTERN = r'\b(?:' + '|'.join(FRACTIONS) + r')\b'
    BASIC_NUMBER_PATTERN = r'\b(?:' + '|'.join(ALL_NUMBER_WORDS) + r')\b'
    PATTERN = f"({FRACTIONS_PATTERN})|({COMPOUND_NUMBER_PATTERN})|({BASIC_NUMBER_PATTERN})"

    def __call__(self, text):
        result = []
        for m in re.finditer(self.PATTERN, text.lower()):
            numstr = m.group(0)
            value = self.text2num(numstr)
            if value is not None:
                result.append({
                    "idx": (m.start(), m.end()),
                    "text": value,
                    "type": "question"
                })
        return result

    def text2num(self, text):
        text = text.lower()
        if text in self.FRACTIONS:
            return self.FRACTIONS[text]
        if text in self.TEENS:
            return str(self.TEENS[text])
        if text in self.TENS:
            return str(self.TENS[text])
        if text in self.UNITS:
            return str(self.UNITS[text])
        match = re.match(r'(%s)[ -](%s)' % ("|".join(self.TENS), "|".join(self.UNITS)), text)
        if match:
            tens, units = match.groups()
            value = self.TENS[tens] + self.UNITS[units]
            return str(value)
        return None
