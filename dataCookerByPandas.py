#!/usr/bin/env python3

import                    pandas              as pd
import                    numpy               as np
import                    re
import                    matplotlib.pyplot   as plt
from   typing      import Any,Union
from   collections import OrderedDict
import                    csv                        # for outargs, pretty output.
import                    sys
import                    argparse

def readDF(input:str, reader:str=None, **kwargs) -> pd.DataFrame:

    fn = pd.read_csv # default reader

    # select reader by ext.
    if   any( input.endswith(suffix) for suffix in ['.xlsx', '.xls']):
       fn = pd.read_excel
    elif any( input.endswith(suffix) for suffix in ['.json', '.js'] ):
       fn = pd.read_json

    if reader is not None:      # when reader specified, use it.
       fn = getattr(pd, reader)

    df = fn(input, **kwargs)    # pd.{read_csv|read_excel|...}(input, ...)
    return df

def writeDF(df:pd.DataFrame, out:str, writer:str=None, **kwargs):
    fn = df.to_csv # default writer

    # select writer by ext.
    if   any( out.endswith(suffix) for suffix in ['.xlsx', '.xls']):
       fn = df.to_excel
    elif any( out.endswith(suffix) for suffix in ['.json', '.js'] ):
       fn = df.to_json

    if writer is not None:      # when writer specified, use it.
       fn = getattr(df, writer)

    fn(out, **kwargs)           # df.{to_csv|to_excel|...}(out, ...)

def getColsFromRow(row:pd.core.series.Series, prefix:str, rmna:bool=False, outas:str='') -> Union[pd.core.series.Series,str] :

    d = row[row.index.str.startswith(prefix)] # pick specified columns in row.
    if rmna:
        d = d.dropna()

    if outas in   [ 'CSV','JoinedStr']:
        rtn = ''
        if len(d)>0:
            d   = d.astype(str) # convert each val type to str
            rtn = ','.join ( d.replace({ 'nan':'null'}) ) # replace nan to null.
        return rtn

    # default case (when outas not specified)
    return d


def describeRow(s:pd.core.series.Series, prefix:str=None, resultName:str=None) -> pd.core.series.Series :
    '''helper function to get describe() for each Row, for df.apply( ..., axis=ROW)

    Args:
       prefix(str):     prefix names of index to pick, before applying describe()
       resultName(str): the name of cell, when merging result into input series

    Returns:
       Series: result of describe itself or merged row.
    '''

    part = s
    if prefix is not None:
        part = s[s.index.str.startswith(prefix)]
    d = part.describe()

    if resultName is None:
        return d

    # merging into input data.
    s[resultName] = d
    return s


def scatter_matrix(df:pd.DataFrame, out:str=None):
    graph = pd.plotting.scatter_matrix(df)
    if out:
       plt.savefig(out)
    plt.show()


#########################################
def str2list(val:Union[str, list]) -> list[Any]:  # deprecated
    '''helper function to get list, when val is str(comma separated or single).'''

    rtn = []

    if isinstance(val,list):
       return val
    if not isinstance(val,str):
       return rtn

    if ',' in val:
        rtn = val.split(',')
    else:
        rtn = [ val ]
    return rtn

def findIdx(expr:str, candidates:pd.Index, startswith:bool=False, regex:bool=False, isin:bool=False) ->list[str]:
    '''helper function to find index(labels of rows or cols) by expr and candidates.'''

    idx = None
    if startswith:
        if candidates is None:
           raise RuntimeError('candidates not specified.')
        idx = candidates [ candidates.str.startswith(expr) ]
        if any(idx):
            return idx.tolist()

    if regex:
        if candidates is None:
           raise RuntimeError('candidates not specified.')
        idx = candidates [ candidates.str.contains(expr, regex=True) ]
        if any(idx):
            return idx.tolist()

    if isin:
        if ',' in expr:
            expr = expr.split(',')
        else:
            expr = [ expr ]
        if candidates is None:
            return expr
        else:
            idx = candidates [ candidates.str.isin(expr) ]
            if any(idx):
               return idx.tolist()
    return idx

def categorizeKWArgs(kwargs: dict[str, Any], nssep:str='::') ->  tuple[ Union[None,dict[str,Any]],  Union[None, dict[str,dict[str,Any]]] ]:
    '''categorize kwargs when keys having namespace( ex, 'ns1:key1').

       kwargs => groupBy namespace => returns pairs of (namespace, dict).
       i.e: try kwargs = { 'simplekey': 'val', 'ns1::key1': 'val1', 'ns2::key2':'val2'}

    Args:
        kwargs(dict[str,Any]): python standard dict which passed by **kwargs
        nssep(st):             namespace separator, default: '::'

    Returns:
        1st item:     None or dict[str,Any].                         partial  dict of kwargs, when no-namespacing keys exists.
        2nd item:     None or dict[ns:str,  dict[subkey:str,Any]]    group-by dict of kwargs, when key has namespace...
    '''
    # phase0) validation...
    if kwargs is None or not any(kwargs):
       return (None,None)

    # phase1) categorize...

    tmp: dict[str, dict[str, Any]] = {}
    for key, val in kwargs.items():
        if nssep in key: # namespace found.
            nsk = key.split(nssep)
            if len(nsk)>2:
               raise RuntimeError('no suppor mutiple namespacing', key)
            ns=nsk[0]
            skey=nsk[1]
        else:            # no namespace.
            ns = nssep   # reserved namespace, which cannot use for key...
            skey = key

        if ns not in tmp:
            tmp[ns] = {}
        tmp[ns][skey] = val

    # phase1 done.
    # phase2) build return value...

    sKV  = None
    nsKV = None
    if nssep in tmp:
       sKV = tmp[nssep]
       del tmp[nssep]

    if any(tmp):
       nsKV = tmp

    return (sKV, nsKV)

class ArgParseKeyValAction(argparse.Action):
    '''argparse helper function to enable key=val style option.

       example to use:
       parser.add_argument('-p', '--params', action=ArgParseKeyValAction,  metavar='KEY:=VAL', sep=':=', help='...')
       tested value type:  str, int, float, bool, None, and more...

       note that delimiter(separator) between key and value is defined as ':=' in the above sample, to keep mathmatic expression as it is.
         
          -p key:=  (no val)...you can set 0 length str
          -p key:=True   => type(val) = boolean
          -p key:=False  => type(val) = boolean
          -p key        => None (by no ':=')
          -p key:=sample => type(val) = str
          -p key:=1      => type(val) = int
          -p dict:='{"key1":True}'    => dict['key1':True]
          -p array:='[1,None,"3"]'    => [1,None,"3"]
          -p tuple:='(1,None,"3")'    => (1,None,"3")
          -p expr:='1<= num <=2'      => "1<= num <=2"
          -p quote:=csv.QUOTE_NONNUMERIC   => 2 in integer or 'csv.QUOTE_NONNUMERIC' ( when it has 'import csv' then integer, else str)
             :
    '''
    def __init__(self, option_strings, dest, nargs='+', sep:Union[str|None]='=', **kwargs):
        '''
        Args:
            sep(str|None): my original param to specify key-value separator. default('=')
        '''
        super().__init__(option_strings, dest, nargs=nargs, **kwargs)
        self.sep=sep #key-value delimiter

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, {})
        for value in values:
            if self.sep in value:
                kv = value.split(self.sep)
                if len(kv)!=2:
                   raise RuntimeError('multiple separator found in parm {value} ==> not supported now...')
                k = kv[0]
                v = kv[1]
                getattr(namespace, self.dest)[k] = self.parseByEval(v)
            else:
                getattr(namespace, self.dest)[value] = None

    @staticmethod
    def parseByEval(v: str) -> Any:
        try:
            rtn = eval(v)
            return rtn
        except (NameError, SyntaxError) as e:
            if v.lower() == 'none':
                return None
            return v

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',  type=str,                    default='/dev/stdin',                     help='path of input')           # input file path
    parser.add_argument('--inargs',       action=ArgParseKeyValAction, default={}, metavar='key:=val', sep=':=', help='extra option for input')  # args for reading
    parser.add_argument('-o', '--output', type=str,                    default='/dev/stdout',                    help='path of output')          # output file path
    parser.add_argument('--outargs',      action=ArgParseKeyValAction, default={}, metavar='key:=val', sep=':=', help='extra option for output') # args for outputing
    parser.add_argument('-m','--myopts',  action=ArgParseKeyValAction, default={}, metavar='key:=val', sep=':=', help='extra option for data processing...')  # options to cook data.
    args = parser.parse_args()
    print(args, file=sys.stderr)


    #if any(':' in k for k in args.myopts.keys()):
    #   myopts = categorizeKWArgs(args.myopts)
    myopts = args.myopts

    # phase 1) read data
    reader=myopts.get('reader', None)
    df = readDF(args.input, reader=reader, **(args.inargs))
    df = df.where(df.notnull(), None) # N/A => None  (cf. df.notnull() == df.notna() == (! df.isna())
    cache = df # cache dataframe, for later use.
    # phase 1) done.

    if 'printfull' in myopts:
        # configure all dataframe would be out at print()
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)

    # phase 2) cooking data        (-m cook=S,Filter,...')
    todo = myopts.get('cook','')
    todo = findIdx(todo, None, isin=True) # just todo.split(',')
    for idx, ops in enumerate( todo ):

        # cook data, according to the request...

        if 'S' == ops:        # task: print summary

            print(df.columns,    file=sys.stderr)
            print(df.shape,      file=sys.stderr)
            print(df.dtypes,     file=sys.stderr)
            # use -o /dev/null to discard content output

        if 'P' == ops:        # task: print

            print(df.describe(), file=sys.stderr)
            print(df.columns,    file=sys.stderr)
            print(df.dtypes,     file=sys.stderr)
            print(df,            file=sys.stderr)

        # common process to data in spreadsheet.<- task:

        if 'Filter' == ops:   # task: pick by label names regex, with df.filter,  dir: cols(OK),rows(OK)     subopts: filter, faxis

            colFilter  = myopts.get('filter',None) # default None
            faxis      = myopts.get('faxis', None) # default None(==1), 0:pickup row, 1:pickup col

            if colFilter is None:
               raise RuntimeError('###### filter param not found, for filter ops')
            #df = df.filter(regex=colFilter, axis=faxis)

            if faxis in [ None, 1, 'columns' ]:
               colFilter = findIdx(colFilter, df.columns, startswith=True,  regex=True, isin=True)
            if faxis in [ 0, 'index' ]:
               colFilter = findIdx(colFilter, df.index,   startswith=False, regex=True, isin=True)

            df = df.filter(items=colFilter, axis=faxis)

        if 'Drop' == ops:     # task: drop by label names,    with df.drop        dir: cols(OK),rows(OK) ( 'NOT' in regex is diffucult...)  subopts:drop, daxis

            dlabel = myopts.get('drop',None)
            daxis  = myopts.get('daxis', None) # default None(col), 0:col, 1:row

            if dlabel is  None:
                raise RuntimeError('###### drop param not found, for Drop ops')

            if daxis in [None, 1, 'columns']:    # drop specified from cols
                dlabel = findIdx(dlabel, df.columns, startsWith=True, regex=True, isin=True)
                df = df.drop(columns=dlabel, axis=1)

            if daxis in [ 0, 'index'       ]:    # drop specified from rows
                dlabel = findIdx(dlabel, df.index, startsWith=True, regex=True, isin=True)
                df = df.drop(labels=dlabel,  axis=0)

        if 'DRow' == ops:     # task: drop by val (=?= pick neq)                  dir: cols(X), rows(OK)        subopts: dcol, dval

            col = myopts.get('dcol','rtt01')
            vals = myopts.get('dval',[])
            df = df[ ~df[col].isin(vals) ] # df whose col ISNOT in vals


        if 'query' == ops:    # task: pick by val(numbers) of colum,              dir: cols(X),rows(OK)   support math expression ( 1<rtt<2 etc). subopts: query

            expr   = myopts.get('query',None)  # query expression, ex: '1<rtt01<2 and 1<rtt02<2'
            if expr is  None:
                raise RuntimeError('###### query param not found, for query ops')
            df = df.query(expr)

        if 'pick' == ops:     # task: pick by val(str) of column,                 dir: cols(X),rows(OK)       subopts: pcol, and more...

            pcol = myopts.get('pcol', None) # column
            exp  = myopts.get('regex',None) # regex search expression
            eq   = myopts.get('eq',   None) # equals to val     (str, single or comma separated)
            neq  = myopts.get('neq',  None) # NOT equals to val (str, single or comma separated)

            if pcol is None:
               raise RuntimeError('###### pcol not found, required param for pick ops')

            if exp is not None:     #subtask: pick rows by val x regex    subopts: regex
                df = df[ df[pcol].apply(lambda x: (bool(re.search(exp, str(x), re.UNICODE))) )]
            if eq  is not None:     #subtask: pick rows by val x equal    subopts: eq
                eq =str2list(eq)
                df = df[ df[pcol].isin(eq) ]
            if neq is not None:     #subtask: pick rows by val x neq      subopts: neq
                neq =str2list(neq)
                df = df[ ~df[pcol].isin(neq) ]

        if 'sort' == ops:     # task: sort rows by specific column                dir: cols(X), rows(OK)    subopts: scol

            scol = myopts.get('scol', None)
            if scol is None:
               raise RuntimeError('###### scol not found, required param for sort ops')
            scol = findIdx(scol, df.columns, startswith=True, regex=True, isin=True)
            df = df.sort_values(by=scol)

        if 'JoinCols' == ops: # task: join colums for each row                    dir: cols(X), rows(OK)   subopts: jcol, jrmna, jout

            jcol =  myopts.get('jcol',     None)
            jrmna = myopts.get('jrmna',    False)
            jout  = myopts.get('jout',     'csv_' + jcol)
            if jcol is None:
               raise RuntimeError('###### required param (jcol) not found, for JoinCol ops')

            s = df.apply(getColsFromRow,  axis=1, prefix=jcol, rmna=jrmna, outas='JoinedStr')
            df [ jout ] = s

        if 'merge' == ops:    # task: merge two dataframes into one (wider)       dir: cols(OK), rows(X), two dataframes    subopts: in2, key

            in2 = myopts.get('in2',  None)
            key = myopts.get('key',  None)
            how = myopts.get('how', 'outer')

            if in2 is None or key is None or how is None:
               raise RuntimeError('###### required param (in2,key,how) not found, for merge ops')

            df2= readDF(in2, reader=None, **(args.inargs))
            if ',' not in key:
               df3 = pd.merge(df, df2, on=key, how=how)
            else:
               key = key.split(',')
               df3 = pd.merge(df, df2, left_on=key[0], right_on=key[1], how=how)
            df = df3

        if 'T' == ops:        # task: transpose (testing)                         dir: swap axis(cols<=>rows)

            df1 = df.T
            #df2 = df1.T
            #print (df2.equals(df)) # TRUEになるには、read時に header=None, index_col=Noneを要指定
            df = df1

        if 'scatter' == ops:  # task: scatter_matrix and save data.                        subopts: gout

            gout = myopts.get('gout', None)
            scatter_matrix(df, out=gout)

        if 'Gsample' == ops: # task: get random samples via groupby               dir: cols(X), rows(OK) subopts: Gcol, samples, picking, seed     NOTE:output ordering of rows will be different than input.

            gcol    = myopts.get('Gcol',     None)    # key for grouping
            samples = myopts.get('samples',  3)       # num of samples.
            picking = myopts.get('picking', 'random') # how to pickup samples
            seed    = myopts.get('seed',     None)    # get randomseed in case of random pickup, to get reproducibility.
            gp = df.groupby(gcol)
            tmp = {}
            for v, g in gp:
                if len(g) <= samples: # when items in group <= samples => select all.
                    tmp[v] = g.copy()
                else:                 # when items in group  > samples => pickup by requests.
                    if picking in ['random', 'r', 'rand']:
                       tmp[v] = g.sample(n=samples, random_state=seed).copy()
                    elif picking in ['head', 'h']:
                       tmp[v] = g.head(samples).copy()
                    elif picking in ['tail', 't']:
                       tmp[v] = g.tail(samples).copy()
            df = pd.concat( tmp.values() )  # mk large DF by cat all samples in rows

        if 'Flaten' == ops:   # task: groupby (f)col, and reform new dataframe    dir: cols(X), rows(o)   subopts: fcol, gcol, prefix

            # one of usecase:  collects raw data(rtt_XXX) for futher groupBy(kyoten) aggregation.
            # sample usage,    -m fcol:=kyoten -gcol:=rtt* prefix:=srtt   --outargs index:=False
            #
            #   NOTE!!!!:  this code generates label in content(col:0) style, to fill A:1 cell in excel/csv. => requires --outargs index:False
            #
            fcol    = myopts.get('fcol',   None) # key of groupby
            gcol    = myopts.get('gcol',   None) # columns to grep (values which fills new dataframe)
            prefix  = myopts.get('prefix', None) # prefix for new column label

            if fcol is None or gcol is None or prefix is None:
               raise RuntimeError('###### required param fcol/gcol/prefix for Flaten ops, like fcol:=kyoten, gcol:=rtt* prefix:=srtt')

            # phase a) get groupName and its occurrences, by df.groupby
            gdf = df.groupby(by=fcol)
            gdf_counts = gdf.size().sort_values(ascending=False)
            gdict = OrderedDict(gdf_counts) # get unique key and its occurrences.

            # phase b) gather data for each group, as input of new dataframe.
            smap:OrderedDict[str,Any]={}
            for key,count in gdict.items():
               g = df[ df[fcol]==key ].filter(regex=gcol, axis=1)
               #va = list( g.values.flatten())
               vflatens = g.values.flatten()
               vflatens = [v for v in vflatens if pd.notnull(v)]
               vflattens= sorted(vflatens)
               #s = pd.Series(vflattens, name=key, index=idx)
               #s = pd.Series(vflattens, name=key)
               s = pd.Series([key]+vflattens)                   # add index in contents
               smap[key]= pd.DataFrame([s]) # smap: 1row dataframe
            # end of phase2

            # phase c) build new dataframe from gathered data.
            df = pd.concat(smap.values(), axis=0) # data part (append in rows)

            # phase c.1) make and re-assign new column labels for pretty-format.
            cl = df.shape[1] -1                                # -1 for index in contents
            ndigits = len(str(cl-1))
            nformat = '{:02d}' if ndigits==1 else '{:0'+str(ndigits)+'d}' # zero-padding
            nformat = prefix + nformat
            idx = [ nformat.format(n) for n in range(1,cl+1) ] # index label completed.
            idx = [fcol] + idx                                 # when index in contents.
            df.columns = idx # assign label in column
            # phase c done.

        # proprietary processing                <- task:

        if 'DR1' == ops:      # task: drop fist row (testing)

            df = df.iloc[1:]

        if 'Groupby' == ops:  # task: pick by val of column and grouping UNDER TESTING

            gcol = myopts.get('gcol',None) # column
            if gcol is None:
               raise RuntimeError('###### gcol param not found, for Groupby ops')
            gdf = df.groupby(by=gcol)
            gdf_counts = gdf.size().sort_values(ascending=False)
            #gdict = OrderedDict(gdf_counts) # get unique key and its occurrences.
            gl = list(zip(gdf_counts.index, gdf_counts))
            gl = list(gp_counts.index) # just index
            print(gl)
            #print(gp.describe())

        # each ops
    # endof for loop => endof cooking data

    if args.output in ['/dev/null', 'null', 'none', None ]:
        exit()

    # phase 3) output data
    writer=myopts.get('writer', None)
    try:
       writeDF(df, args.output, writer=writer, **(args.outargs))
    except BrokenPipeError as e:
       print("BrokenPipeError detected then ignore...", file=sys.stderr)
       pass # this occures when output==/dev/stdout and piping to  more/head/tail...

    #end

# behaviour in parameters in read/write
#  read                    write
#  (pd.read_xxx)          (df.to_xxx)
# ---------------------------------
#  (default)              (default)   => extra/dusty index added in each row.
#  (default)              index=False => as expected    i.e requires --outargs index:=False
#  index_col='A'          (default)   => as expected    i.e requires --inargs  index_col:='A'
#
# other opts to get better output
#  CSV   --outargs quoting:=csv.QUOTE_NONNUMERIC        output to CSV
#  JSON  --outargs force_ascii:=False orient:=records   output to json

# at merge...
#  -i ./inA.xlsx -m cook:=merge in2:=./inB.csv key:=colA,colB -o ./out.xlsx --outargs index:=False
#
#----
#
#  BASIS of axis in pandas,  in MOST cases.     df.filter is somewhat different !?
#
#  val of axis    | process direction | input 'series' to function
#  ---------------+-------------------+-----------------------------
#  None,0,columns | col by col        | all rows   in specific col
#  1,index        | row by row        | all fields in specific row
