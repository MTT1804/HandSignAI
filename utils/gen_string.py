import random
#Genruje losowy string o długości 1000 z liter podanych w liście letters
def generate_string_of_length_1000():
    letters = ['A', 'B', 'C', 'D', 'E']
    result = ''.join(random.choice(letters) for _ in range(1000))
    return result

random_string = generate_string_of_length_1000()
print(random_string)
