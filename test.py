from pymongo import MongoClient

# Coloque a string direto no código
MONGO_URI = "mongodb+srv://isabellapilasilva_db_user:PpenvfPqPlM0NND6@cluster0.mzze1os.mongodb.net/infobot?retryWrites=true&w=majority&tls=true"

client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10000)

try:
    print(client.server_info())
    print("✅ Conexão com o MongoDB Atlas estabelecida com sucesso!")
except Exception as e:
    print("❌ Erro ao conectar:", e)
