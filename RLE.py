def RLE(symb_index,K):
    symbols = []
    #flatten symb_index matrix to a vector
    for i in range(len(symb_index)):
        symbols.extend(symb_index[i])
    run_symbols = []
    #loop through quantized elements, if it is zero increment zero_counter, else add the symbol and the number of leading zeros to the nx2 table 
    zero_counter = 0
    for i in range(K):
        if int(symbols[i]) == 0 :
            zero_counter += 1
        else:
            run_symbols.append([int(symbols[i]),zero_counter])
            zero_counter = 0
    #in the end add any zeros that are left (if the array of quantized elements ends with 0)
    if zero_counter != 0:
        run_symbols.append([0,zero_counter-1])
    return run_symbols

def iRLE(run_symbols,K):
    symb_index = []
    #for every row of the rle table add the amount of leading zeros and then the non-zero element
    for r in run_symbols:
        if r[1] != 0: #no need to add 0 if there are 0 leading zeros
            symb_index.extend(r[1]*[0])
        symb_index.append(r[0])
    return symb_index