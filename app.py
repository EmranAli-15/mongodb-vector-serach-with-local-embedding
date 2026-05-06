from flask import Flask, request
import os
import json
from flask import jsonify
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from sentence_transformers import SentenceTransformer


import pymongo


app = Flask(__name__)

# --- CONFIGURATION & GLOBAL INITIALIZATION ---
DB_PATH = "./book_db"
# MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_NAME = "intfloat/multilingual-e5-small"

# Load model globally so it's only loaded once into memory
embeddings_model = HuggingFaceEmbeddings(model_name=MODEL_NAME)



mongo_client = pymongo.MongoClient("mongodb+srv://BookWorld:1k9x4ClxqAlewBWZ@cluster0.2b4mnlf.mongodb.net/BookWorld?retryWrites=true&w=majority&appName=Cluster0")
db = mongo_client["BookWorld"]
collection = db["vector-data"]


@app.route("/")
def index():
    return "Hello"

@app.route("/embedding")
def embedding():
    full_text = """
আমি/আমারা কারা? : আমি বুক ওয়ার্ল্ড এজেন্ট, আপনার কোন সাহায্য লাগলে জানাবেন।
About us or who me/we: I’m book world agent, let me know if you need any help.
যোগাযোগ (Contact Information)/আমাদের সাথে কিভাবে যোগাযোগ করবেন: বুক ওয়ার্ল্ড (Book World) কাস্টমার কেয়ারের সাথে মোবাইলে যোগাযোগ করতে কল করুন ০১৯৮২৭৪০০৬৪ নাম্বারে। ইমেইলের মাধ্যমে যোগাযোগ করতে চাইলে মেইল করুন bookworld-service@gmail.com এই ঠিকানায়। আমাদের অফিসের ঠিকানা: সাতাইশ, টঙ্গী, গাজীপুর। কাস্টমার সাপোর্ট সংক্রান্ত যেকোনো প্রয়োজনে আমাদের সাথে এই মাধ্যমগুলোতে যোগাযোগ করা যাবে।
Contact information: To contact book world customer care, call to the number 01982740064. Send a main to bookworld-service@gmail.com to contact throw email. Our office location is Sataish, Tongi, Gazipur. You can contact following these septs for any customer supports.
ডেলিভারি চার্জ এবং সময় /আমাদের ডেলিভারি চার্জ কত: ঢাকা সিটির ভেতরে ডেলিভারি চার্জ ৬০ টাকা এবং পণ্য হাতে পেতে ২-৩ দিন সময় লাগে। ঢাকা সিটির বাইরে ডেলিভারি চার্জ ১২০ টাকা এবং সময় লাগে ৩-৫ দিন। আমরা ডেলিভারির জন্য পাঠাও কুরিয়ার সার্ভিস ব্যবহার করে থাকি।
Delivery charge and time: Our delivery charge inside Dhaka city is 60tk and needs 2-3 days to hand delivery. Delivery charge outside Dhaka city is 100tk and needs 3-5 days to hand delivery. For delivery we use ‘Pathao Courier’ service
রিটার্ন পলিসি এবং নিয়মাবলী (Return Policy)/রিটার্ন পলিসি কি: বই রিটার্ন বা ফেরত দেওয়ার নিয়ম: ডেলিভারি করা বইটি যদি আপনার অর্ডার করা বই না হয় অথবা বইটির গুণগত মান (Quality) খারাপ থাকে, তবে বই হাতে পাওয়ার ৩ দিনের মধ্যে ০১৯৮২৭৪০০৬৪ নাম্বারে যোগাযোগ করতে হবে রিটার্ন (return) করার জন্য। তবে খেয়াল রাখতে হবে যেন আপনার মাধ্যমে বইটির কোনো ক্ষতি না হয়। রিটার্ন করা প্রোডাক্ট স্টকে থাকলে ৩-৫ দিনের মধ্যে নতুন বই পাঠানো হবে এবং এর জন্য অতিরিক্ত কোনো ডেলিভারি চার্জ দিতে হবে না।
Return policy and book/product return rules: If the book is not you ordered or the book quality is bad, then you should contact with book world customer care within 3 days from delivered day. If the returned product/book in our stock then you will get a new copy within 3-5 days and don’t need to pay any more delivery charge.
রিফান্ড বা টাকা ফেরত (Refund Policy)/কিভাবে রিফান্ড বা টাকা ফেরত পাবেন: পণ্য বা বই ফিরিয়ে দিলে রিফান্ডের জন্য ডেলিভারি চার্জ বাদে বাকি সব টাকা ফেরত (Refund/ রিফান্ড) দেওয়া হবে। টাকা ফেরত/ রিফান্ড নেওয়ার পদ্ধতি সম্পর্কে আমাদের টিম আপনার সাথে সরাসরি যোগাযোগ করবে।
Refund policy: For return a product/book we will refund full 95% price of the product/book except delivery charge. To send back refund, our team will contact with you.
পেমেন্ট মেথড বা পেমেন্ট পদ্ধতি (Payment Methods)/আমাদের পেমেন্ট মেথড/কিভাবে পেমেন্ট নিয়ে থাকি: আমাদের সাইটে কেনাকাটার জন্য পেমেন্ট মেথড হিসেবে ১. ক্যাশ অন ডেলিভারি (Cash on Delivery) এবং ২. বিকাশ পেমেন্ট (Bkash Payment) সিস্টেম চালু আছে।
Our payment methods: We offer two ways to payment us, one is ‘cash on delivery’ and another is ‘bkash payment’.

বইয়ের তালিকা ও মূল্য/দাম (Books List & Price)
আমাদের কাছে শাহরিয়ার খান-এর লেখা বেসিক আলী-3 বইটি আছে, বেসিক আলী-3 বইটির মূল্য বা দাম (মূল্যঃ 140 টাকা)। We have the Basic Ali-3 book by-Shahriar Khan writer, price of Basic Ali-3 book is 140

আমাদের কাছে জহির রায়হান-এর লেখা কয়েকটি মৃত্যু বইটি আছে, কয়েকটি মৃত্যু বইটির মূল্য বা দাম (মূল্যঃ 155 টাকা)। We have the Koyekti Mrityu book by-Zahir Raihan writer, price of Koyekti Mrityu book is 155

আমাদের কাছে কি কেন কিভাবে-এর লেখা বিজ্ঞানের বিস্ময় - ভলিউম ১ (হার্ডকভার) বইটি আছে, বিজ্ঞানের বিস্ময় - ভলিউম ১ (হার্ডকভার) বইটির মূল্য বা দাম (মূল্যঃ 120 টাকা)। We have the Bigganer Bismoy - Volume 1 (Hardcover) book by-Ki Keno Kibhabe writer, price of Bigganer Bismoy - Volume 1 (Hardcover) book is 120

আমাদের কাছে শাহরিয়ার খান-এর লেখা বেসিক আলী-4 বইটি আছে, বেসিক আলী-4 বইটির মূল্য বা দাম (মূল্যঃ 150 টাকা)। We have the Basic Ali-4 book by-Shahriar Khan writer, price of Basic Ali-4 book is 150

আমাদের কাছে হুমায়ূন আহমেদ-এর লেখা নবনী বইটি আছে, নবনী বইটির মূল্য বা দাম (মূল্যঃ 241 টাকা)। We have the Noboni book by-Humayun Ahmed writer, price of Noboni book is 241

আমাদের কাছে তামিম শাহরিয়ার সুবিন-এর লেখা পাইথন দিয়ে প্রোগ্রামিং শেখা বইটি আছে, পাইথন দিয়ে প্রোগ্রামিং শেখা বইটির মূল্য বা দাম (মূল্যঃ 215 টাকা)। We have the Python Diye Programming Shekha book by-Tamim Shahriar Subeen writer, price of Python Diye Programming Shekha book is 215

আমাদের কাছে শাহরিয়ার খান-এর লেখা বেসিক আলী-১ বইটি আছে, বেসিক আলী-১ বইটির মূল্য বা দাম (মূল্যঃ 120 টাকা)। We have the Basic Ali-1 book by-Shahriar Khan writer, price of Basic Ali-1 book is 120

আমাদের কাছে মুহম্মদ জাফর ইকবাল-এর লেখা থিওরি অফ রিলেটিভিটি বইটি আছে, থিওরি অফ রিলেটিভিটি বইটির মূল্য বা দাম (মূল্যঃ 180 টাকা)। We have the Theory of Relativity book by-Muhammed Zafar Iqbal writer, price of Theory of Relativity book is 180

আমাদের কাছে কি কেন কিভাবে-এর লেখা The Why? Magazine - August 2025 Edition বইটি আছে, The Why? Magazine - August 2025 Edition বইটির মূল্য বা দাম (মূল্যঃ 120 টাকা)। We have the The Why? Magazine - August 2025 Edition book by-Ki Keno Kibhabe writer, price of The Why? Magazine - August 2025 Edition book is 120

আমাদের কাছে তামিম শাহরিয়ার সুবিন-এর লেখা কম্পিউটার প্রোগ্রামিং-প্রথম খণ্ড বইটি আছে, কম্পিউটার প্রোগ্রামিং-প্রথম খণ্ড বইটির মূল্য বা দাম (মূল্যঃ 155 টাকা)। We have the Computer Programming-Prothom Khondo book by-Tamim Shahriar Subeen writer, price of Computer Programming-Prothom Khondo book is 155

আমাদের কাছে হুমায়ূন আহমেদ-এর লেখা তোমাকে বইটি আছে, তোমাকে বইটির মূল্য বা দাম (মূল্যঃ 180 টাকা)। We have the Tomake book by-Humayun Ahmed writer, price of Tomake book is 180

আমাদের কাছে মুহম্মদ জাফর ইকবাল-এর লেখা বিগ ব্যাং থেকে হোমো স্যাপিয়েনস বইটি আছে, বিগ ব্যাং থেকে হোমো স্যাপিয়েনস বইটির মূল্য বা দাম (মূল্যঃ 130 টাকা)। We have the Big Bang theke Homo Sapiens book by-Muhammed Zafar Iqbal writer, price of Big Bang theke Homo Sapiens book is 130

আমাদের কাছে বিভূতিভূষণ বন্দ্যোপাধ্যায়-এর লেখা আরণ্যক বইটি আছে, আরণ্যক বইটির মূল্য বা দাম (মূল্যঃ 200 টাকা)। We have the Aranyak book by-Bibhutibhushan Bandyopadhyay writer, price of Aranyak book is 200

আমাদের কাছে শাহরিয়ার খান-এর লেখা বেসিক আলী-2 বইটি আছে, বেসিক আলী-2 বইটির মূল্য বা দাম (মূল্যঃ 200 টাকা)। We have the Basic Ali-2 book by-Shahriar Khan writer, price of Basic Ali-2 book is 200

আমাদের কাছে তামিম শাহরিয়ার সুবিন-এর লেখা কম্পিউটার প্রোগ্রামিং ৩য় খণ্ড : ডেটা স্ট্রাকচার ও অ্যালগরিদম বইটি আছে, কম্পিউটার প্রোগ্রামিং ৩য় খণ্ড : ডেটা স্ট্রাকচার ও অ্যালগরিদম বইটির মূল্য বা দাম (মূল্যঃ 280 টাকা)। We have the Computer Programming 3rd Khondo : Data Structure o Algorithm book by-Tamim Shahriar Subeen writer, price of Computer Programming 3rd Khondo : Data Structure o Algorithm book is 280

আমাদের কাছে মুহম্মদ জাফর ইকবাল-এর লেখা রহস্যময় ব্ল্যাক হোল বইটি আছে, রহস্যময় ব্ল্যাক হোল বইটির মূল্য বা দাম (মূল্যঃ 130 টাকা)। We have the Rohoshomoy Black Hole book by-Muhammed Zafar Iqbal writer, price of Rohoshomoy Black Hole book is 130

আমাদের কাছে তামিম শাহরিয়ার সুবিন-এর লেখা পাইথন দিয়ে প্রোগ্রামিং শেখা ২য় খণ্ড - অবজেক্ট ওরিয়েন্টেড প্রোগ্রামিং ও ওয়েব ক্রলিং বইটি আছে, পাইথন দিয়ে প্রোগ্রামিং শেখা ২য় খণ্ড - অবজেক্ট ওরিয়েন্টেড প্রোগ্রামিং ও ওয়েব ক্রলিং বইটির মূল্য বা দাম (মূল্যঃ 259 টাকা)। We have the Python Diye Programming Shekha 2nd Khondo - Object Oriented Programming o Web Crawling book by-Tamim Shahriar Subeen writer, price of Python Diye Programming Shekha 2nd Khondo - Object Oriented Programming o Web Crawling book is 259

আমাদের কাছে শাহরিয়ার খান-এর লেখা বেসিক আলী-5 বইটি আছে, বেসিক আলী-5 বইটির মূল্য বা দাম (মূল্যঃ 130 টাকা)। We have the Basic Ali-5 book by-Shahriar Khan writer, price of Basic Ali-5 book is 130

আমাদের কাছে বিভূতিভূষণ বন্দ্যোপাধ্যায়-এর লেখা চাঁদের পাহাড় বইটি আছে, চাঁদের পাহাড় বইটির মূল্য বা দাম (মূল্যঃ 180 টাকা)। We have the Chander Pahar book by-Bibhutibhushan Bandyopadhyay writer, price of Chander Pahar book is 180

আমাদের কাছে কি কেন কিভাবে-এর লেখা Divention (বিজ্ঞান সাময়িকী) June 2025 Edition বইটি আছে, Divention (বিজ্ঞান সাময়িকী) June 2025 Edition বইটির মূল্য বা দাম (মূল্যঃ 120 টাকা)। We have the Divention (Biggan Samoyiki) June 2025 Edition book by-Ki Keno Kibhabe writer, price of Divention (Biggan Samoyiki) June 2025 Edition book is 120

আমাদের কাছে হুমায়ূন আহমেদ-এর লেখা অপেক্ষা বইটি আছে, অপেক্ষা বইটির মূল্য বা দাম (মূল্যঃ 320 টাকা)। We have the Opekkha book by-Humayun Ahmed writer, price of Opekkha book is 320

আমাদের কাছে বিভূতিভূষণ বন্দ্যোপাধ্যায়-এর লেখা পথের পাঁচালী বইটি আছে, পথের পাঁচালী বইটির মূল্য বা দাম (মূল্যঃ 190 টাকা)। We have the Pother Panchali book by-Bibhutibhushan Bandyopadhyay writer, price of Pother Panchali book is 190

আমাদের কাছে তামিম শাহরিয়ার সুবিন-এর লেখা কম্পিউটার প্রোগ্রামিং - দ্বিতীয় খণ্ড বইটি আছে, কম্পিউটার প্রোগ্রামিং - দ্বিতীয় খণ্ড বইটির মূল্য বা দাম (মূল্যঃ 292 টাকা)। We have the Computer Programming - Dwitiyo Khondo book by-Tamim Shahriar Subeen writer, price of Computer Programming - Dwitiyo Khondo book is 292

আমাদের কাছে জহির রায়হান-এর লেখা হাজার বছর ধরে বইটি আছে, হাজার বছর ধরে বইটির মূল্য বা দাম (মূল্যঃ 155 টাকা)। We have the Hajar Bochor Dhore book by-Zahir Raihan writer, price of Hajar Bochor Dhore book is 155

আমাদের কাছে জহির রায়হান-এর লেখা শেষ বিকেলের মেয়ে বইটি আছে, শেষ বিকেলের মেয়ে বইটির মূল্য বা দাম (মূল্যঃ 172 টাকা)। We have the Shesh Bikeler Meye book by-Zahir Raihan writer, price of Shesh Bikeler Meye book is 172

আমাদের কাছে মুহম্মদ জাফর ইকবাল-এর লেখা দেখা আলো না দেখা রূপ বইটি আছে, দেখা আলো না দেখা রূপ বইটির মূল্য বা দাম (মূল্যঃ 130 টাকা)। We have the Dekha Alo Na Dekha Rup book by-Muhammed Zafar Iqbal writer, price of Dekha Alo Na Dekha Rup book is 130

আমাদের কাছে হুমায়ূন আহমেদ-এর লেখা কৃষ্ণপক্ষ বইটি আছে, কৃষ্ণপক্ষ বইটির মূল্য বা দাম (মূল্যঃ 199 টাকা)। We have the Krishnopokkho book by-Humayun Ahmed writer, price of Krishnopokkho book is 199

আমাদের কাছে হুমায়ূন আহমেদ-এর লেখা দেয়াল বইটি আছে, দেয়াল বইটির মূল্য বা দাম (মূল্যঃ 430 টাকা)। We have the Deyal book by-Humayun Ahmed writer, price of Deyal book is 430

আমাদের কাছে কি কেন কিভাবে-এর লেখা The Why? Magazine - May 2025 Edition বইটি আছে, The Why? Magazine - May 2025 Edition বইটির মূল্য বা দাম (মূল্যঃ 120 টাকা)। We have the The Why? Magazine - May 2025 Edition book by-Ki Keno Kibhabe writer, price of The Why? Magazine - May 2025 Edition book is 120
"""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=40,
        is_separator_regex=False
    )

    texts = splitter.split_text(full_text)


    # For storing local db (chromadb)
    chunks = [Document(page_content=t) for t in texts]
    if os.path.exists(DB_PATH):
        vectorstore = Chroma(
            persist_directory=DB_PATH,
            embedding_function=embeddings_model
        )
    else:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings_model,
            persist_directory=DB_PATH
        )

    
    # For store in mongodb search index
    documents_to_insert = []
    for chunk in texts:
        embedding = embeddings_model.embed_query(f"{chunk}")

        doc = {
            "text": chunk,
            "embedding": embedding
        }
        documents_to_insert.append(doc)
    if documents_to_insert:
        result = collection.insert_many(documents_to_insert)
        print(f"সফলভাবে {len(result.inserted_ids)}টি ডকুমেন্ট ইনসার্ট করা হয়েছে।")
    else:
        print("ইনসার্ট করার মতো কোনো ডাটা পাওয়া যায়নি।")

    return f"{len(chunks)}\n\n {texts}"



model = SentenceTransformer('intfloat/multilingual-e5-small')
@app.route('/get/<text>')
def get(text):

    query_embedding = model.encode(f"query: {text}").tolist()

    pipeline = [
        {
            "$vectorSearch":{
                "index":"vector_index",
                "path":"embedding",
                "queryVector":query_embedding,
                "numCandidates": 100,
                "limit":5
            }
        },
        {
            "$project":{
                "_id": 0,
                "text": 1,
                #"score": {"$meta": "vectorSearchScore"}
            }
        }
    ]

    result = list(collection.aggregate(pipeline))
    return f"{result}"



@app.route("/text/<ask>")
def text(ask):
    query_text = ask

    pipeline = [
        {
            "$search":{
                "index":"default",
                "text":{
                    "query": query_text,
                    "path": "text"
                }
            }
        },
        {
            "$limit": 5
        },
        {
            "$project":{
                "_id": 0,
                "text": 1,
                #"score": {"$meta": "vectorSearchScore"}
            }
        }
    ]

    result = list(collection.aggregate(pipeline))
    return f"{result}"




@app.route("/hybrid/<ask>")
def hybrid(ask):
    query_text = ask
    query_vector = model.encode(f"{ask}").tolist()

    vector_results = list(collection.aggregate([
        {
            "$vectorSearch":{
                "index": "vector_index",
                "path":"embedding",
                "queryVector": query_vector,
                "numCandidates": 50,
                "limit": 5
            }
        },
        {
            "$project": {
                "text": 1,
                "_id": 1
            }
        }
    ]))


    text_results = list(collection.aggregate([
        {
            "$search":{
                "index": "default",
                "text": {
                    "query": query_text,
                    "path": "text"
                }
            }
        },
        {
            "$limit": 5
        },
        {
            "$project": {
                "text": 1,
                "_id": 1
            }
        }
    ]))

    rrf_map = {}
    k = 60

    # Process Vector Results
    for rank, doc in enumerate(vector_results, 1):
        doc_id = str(doc['_id'])
        # If ID isn't in map, initialize it
        if doc_id not in rrf_map:
            rrf_map[doc_id] = {"score": 0, "text": doc['text']}
    
        # Add to the score
        rrf_map[doc_id]["score"] += (1 / (k + rank))

    # Process Text Results
    for rank, doc in enumerate(text_results, 1):
        doc_id = str(doc['_id'])
        # If ID isn't in map (not found by vector), initialize it
        if doc_id not in rrf_map:
            rrf_map[doc_id] = {"score": 0, "text": doc['text']}
    
        # Add to the score (this creates the "Hybrid" boost for matches in both)
        rrf_map[doc_id]["score"] += (1 / (k + rank))

    # Convert dictionary to a list and sort by score descending
    sorted_results = sorted(rrf_map.values(), key=lambda x: x['score'], reverse=True)

    # Final top 5 for your RAG prompt
    top_5_text = [item['text'] for item in sorted_results[:5]]

    # Return as clean JSON
    return f"{top_5_text}"




@app.route("/ask/<text>")
def ask(text):
    query_text = text
    client = chromadb.PersistentClient(path=DB_PATH)
    
    try:
        collection = client.get_collection(name="langchain")
        query_vector = embeddings_model.embed_query(query_text)
        
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=5,
        )

        # রেজাল্ট থেকে শুধু টেক্সট অংশটুকু নিয়ে আসা
        documents = results.get('documents', [[]])[0]
        
        # যদি ব্রাউজারে বাংলা দেখতে চান:
        if not documents:
            return "দুঃখিত, কোনো তথ্য পাওয়া যায়নি।"
        
        # সবকটি রেজাল্টকে দাড়ি দিয়ে জোড়া লাগিয়ে রিটার্ন করা
        return "<br><br>".join(documents) 
    
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    app.run(debug=True)