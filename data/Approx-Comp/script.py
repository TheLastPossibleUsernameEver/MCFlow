import csv
with open ("processed_s20071.csv", "r") as source:
    rdr = csv.reader(source)
    with open("processed_s20071_2.csv","w") as result:
        wtr = csv.writer(result)
        for r in rdr:
            wtr.writerow((r[2], r[3]))
