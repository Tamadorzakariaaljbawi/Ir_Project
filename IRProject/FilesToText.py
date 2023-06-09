with open("corpus1/CISI/CISI.ALL") as f:
 lines=""
 for l in f.readlines():
      lines += "\n" +l.strip() if l.startswith(".") else " " + l.strip()
 lines = lines.lstrip("\n").split("\n")
 n=1
for l in lines[:n]:
    print(l)

    print("===================================================")
doc_set = {}
doc_id = ""
doc_auther = ""
doc_text = ""

for l in lines:
    if l.startswith(".I"):
        doc_id = l.split(" ")[1].strip()

    elif l.startswith(".X"):
        doc_set[doc_id] = doc_text.lstrip(" ")
        with open("corpus1/"+ doc_id+".text", 'w') as f:
            f.write(doc_set[doc_id])
        doc_id = ""
        doc_text = ""
    else:
        doc_text += l.strip()[3:]