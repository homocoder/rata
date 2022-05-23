# %%
def is_float(element) -> bool:
    try:
        float(element)
        return True
    except ValueError:
        return False

def is_int(element) -> bool:
    try:
        int(element)
        return True
    except ValueError:
        return False

def is_isoformat(element) -> bool:
    from datetime import datetime
    try:
        datetime.fromisoformat(element)
        return True
    except ValueError:
        return False

# %%
def parse_argv(argv):
    from datetime import datetime
    _conf = dict()
    for i in argv[1:]:
        if '=' in i:
            param = i.split('=')
            _conf[param[0].replace('--', '')] = param[1]

    for i in _conf:
        b = _conf[i]
        if   b == 'True':
            _conf[i] = True
        elif b == 'False':
            _conf[i] = False
        elif is_int(b):
            _conf[i] = int(b)
        elif is_float(b):
            _conf[i] = float(b)
        elif is_isoformat(b):
            _conf[i] = datetime.fromisoformat(b)
    return _conf

def load_conf(file_name):
    from json import load
    fd = open(file_name, 'rt')
    conf = load(fd)
    fd.close()
    return conf

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
    from numpy import array
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out-1
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

def lstm_prep(X, y, n_steps_in=9, n_steps_out=1):
    import numpy as np
    nsamples = len(X)
    ncolumns = len(X.columns)
    # choose a number of time steps
    #n_steps_in, n_steps_out = 90, 1

    # multivariate multi-step data preparation

    # define input sequence # convert to [rows, columns] structure

    in_seq = list()
    for i in range(ncolumns):
        in_seq.append(X.iloc[:, i].values.reshape((nsamples, 1))) ### X here

    out_seq = y    ### y here
    out_seq = out_seq.values.reshape((len(out_seq), 1))

    # horizontally stack columns
    in_seq.append(out_seq)
    dataset = np.hstack(tuple(in_seq)) # here, in_seq contains in_seq and out_seq, just for var economy

    # covert into input/output
    XX, YY = split_sequences(dataset, n_steps_in, n_steps_out)
    print(XX.shape, YY.shape)

    return XX, YY

def sort_human(l):
    import re
    convert = lambda text: float(text) if text.isdigit() else text
    alphanum = lambda key: [convert(c) for c in re.split('([-+]?[0-9]*\.?[0-9]*)', key)]
    l.sort(key=alphanum)
    return l

def copy_from_stringio(conn, df, table):
    """
    Here we are going save the dataframe in memory 
    and use copy_from() to copy it to the table
    """
    from io import StringIO
    import psycopg2
    # save dataframe to an in memory buffer
    buffer = StringIO()
    df.to_csv(buffer, index=False, header=False)
    buffer.seek(0)
    
    cursor = conn.cursor()
    try:
        cursor.copy_from(buffer, table, sep=",")
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        conn.rollback()
        cursor.close()
        return 1
    print("copy_from_stringio() done")
    cursor.close()