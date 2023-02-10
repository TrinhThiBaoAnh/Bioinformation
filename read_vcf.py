def run(path):
    import os
    from glob import glob
    vcf_list = glob(os.path.join(path,"*.vcf" ))
    print(os.path.join(path,"/*.vcf" ))
    my_rs = []
    for vcf in vcf_list:
        print(vcf)
        f = open(vcf,"r")
        fo = open(vcf.replace(".vcf",".txt").replace("GRCh38","outputs"),"w")
        lines = f.readlines()
        for line in lines:
            if ("rs" in line):
                lst = line.split("\t")
                for i in lst:
                    if ("rs" in i and len(i) < 13):
                        # print(i)
                        my_rs.append(i)
                        fo.write(i+"\n")
        f.close()
        fo.close()

    set_rs = set(my_rs)
    print("Len of list rs: ", len(my_rs))
    print("Len of set rs: ", len(set_rs))
    num_features = len(set_rs)
    foo = open("features.txt","w")
    for rs in set_rs:
        foo.write(rs +"\n")
    foo.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Create a set of features from *.vcf files')
    parser.add_argument('--path', type=str, 
                        help='Path to vcf folder')
    args = parser.parse_args()
    print(args.path)
    run(args.path)