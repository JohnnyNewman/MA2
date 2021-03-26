# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 14:38:03 2021

@author: Nils
"""

import os
import fnmatch
import h5py
import pandas as pd
import sys


def process_config(dir, fname):
    with open(os.path.join(dir, fname)) as f:
        lines = f.readlines()
        #print(lines)
        df = pd.DataFrame()
        for l in lines:
            #try:
            ls = l.split("=", 1)
            k = ls[0].strip()
            if k.startswith("%"):
                continue
            v = ls[1].strip()
            #print(k,v)
            df[k] = pd.Series([v], dtype="str")
    return df, True

def process_filename(dir, fname, last_readtime):
    file_modtime = os.stat(os.path.join(dir, fname)).st_mtime
    #print("file_modtime:", datetime.fromtimestamp(file_modtime))
    
    if file_modtime > last_readtime:
        if fname == "surface_flow.csv":
            return pd.read_csv(os.path.join(dir, fname)), True
        elif fname == "surface_adjoint.csv":
            return pd.read_csv(os.path.join(dir, fname)), True
        elif fname == "history_direct.csv":
            return pd.read_csv(os.path.join(dir, fname)), True
        elif fname == "history_adjoint.csv":
            return pd.read_csv(os.path.join(dir, fname)), True
        elif fname == "of_grad_cd.csv":
            return pd.read_csv(os.path.join(dir, fname)), True
        elif fnmatch.fnmatch(fname, "config_*.cfg"):
            return process_config(dir, fname)
    else:
        #print("file_modtime < last_readtime", datetime.fromtimestamp(file_modtime), datetime.fromtimestamp(last_readtime))
        pass
    
    return None, False



def main(h5_filename, cwd="."):
    with h5py.File('mytestfile.hdf5', 'a', libver='latest') as db:
        
        db.swmr_mode = True
    
        for rootdir, dirs, files in os.walk(cwd):
            print (rootdir, dirs)
            #print(files)
            #print()
    
            for fn in files:
                grp_name = str(os.path.join(os.path.relpath(rootdir, start=cwd), fn)).replace("\\", "/")
                #print("grp_name", grp_name, grp_name in db)
                last_readtime = 0
                if grp_name in db:
                    #print("grp_name", grp_name)
                    if "readtime" in db[grp_name].attrs.keys():
                        last_readtime = db[grp_name].attrs["readtime"]
                        #print("last_readtime:", datetime.fromtimestamp(last_readtime))
                ret, ok = process_filename(rootdir, fn, last_readtime)
                if ok:
                    if grp_name in db:
                        del db[grp_name]
                    grp = db.create_group(grp_name)
                    print("  +", grp_name)
                    #print("created grp_name", grp_name, grp_name in db)
                    grp.attrs["readtime"] = datetime.now().timestamp()
                    #if not rootdir in db.keys():
                    #    grp = db.create_group(rootdir)
                    #else:
                    #    grp = db[rootdir]
                    for col in ret.columns:
                        #grp.create_dataset()
                        grp_name = str(os.path.join(os.path.relpath(rootdir, start=cwd), fn, col.replace("\"", "").strip())).replace("\\", "/")
                        #print(grp_name)
                        print("  +", grp_name)
                        if grp_name in db:
                            db[grp_name] = ret[col]
                        else:
                            if ret[col].dtype == "O":
                                db.create_dataset(grp_name, data=str(ret[col]))
                            else:
                                db.create_dataset(grp_name, data=ret[col])
                    print()
                    db.flush()
            print()

if __name__ == "__main__":
    h5_filename = "results.h5"
    cwd = "."
    if len(sys.argv) >= 2:
        h5_filename = sys.argv[1]
    if len(sys.argv) >= 3:
        cwd = sys.argv[2]
    print("h5_filename:", h5_filename)
    print("cwd:", cwd)
    
    main(h5_filename, cwd)