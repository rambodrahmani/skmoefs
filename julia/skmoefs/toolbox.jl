"""
    SK-MOEFS Toolbox.
"""

milestones = [500, 1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000, 75000, 100000]

function load_dataset(name::String)
    """
    Loads the dataset identified by the given name.

    `name` dataset name
    :return:
        X: NumPy matrix NxM (N: number of samples; M: number of features) representing input data
        y: Numpy vector Nx1 representing the output data
        attributes: range of values in the format [min, max] for each feature
        input: names of the features
        output: name of the outputs
    """
    attributes = []
    inputs = []
    outputs = []
    X = []
    y = []
    with open('../dataset/' + name + '.dat', 'r') as f:
        line = f.readline()
        while line:
            if line.startswith("@"):
                txt = line.split()
                if txt[0] == "@attribute":
                    domain = re.search('(\[|\{)(.+)(\]|\})', line)
                    attributes.append(eval('[' + domain.group(2) + ']'))
                elif txt[0] == "@inputs":
                    for i in range(len(txt) - 1):
                        inputs.append(txt[i + 1].replace(',', ''))
                elif txt[0] == "@outputs":
                    outputs.append(txt[1])
            else:
                row = eval('[' + line + ']')
                if len(row) != 0:
                    X.append(row[:-1])
                    y.append(row[-1])
            line = f.readline()
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)
    return X, y, attributes, inputs, outputs
end