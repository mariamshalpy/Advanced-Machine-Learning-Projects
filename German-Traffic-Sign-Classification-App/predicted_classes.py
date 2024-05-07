def class_prediction():
    # Open the file in read mode
    with open('signs.txt', 'r') as file:
        # Initialize an empty dictionary to store the lines
        lines_dict = {}
        
        # Read each line in the file
        for index, line in enumerate(file):
            # Remove any trailing newline characters and add the line to the dictionary
            lines_dict[index] = line.strip()

    return lines_dict 

def get_class_name(predicted_number):
    classes_dict = class_prediction()
    return classes_dict[predicted_number]   
