from flask import Flask, render_template, request
import tensorflow as tf
import pandas as pd

app = Flask(__name__)

loaded_model = tf.keras.models.load_model('./model/mcc_model.h5')
print("Model is loaded")

loaded_model.summary()


colList = ['5up', 'Antfrost', 'Asnazum', 'BBPaws', 'BadBoyHalo', 'Bitzel', 'Calvin', 'CaptainPuffy', 'CaptainSparklez', 'ConnorEatsPants', 'Cubfan135', 'DanTDM', 'DethRidge', 'DrGluon', 'Dream', 'Eret', 'F1NN5TER', 'FalseSymmetry', 'FlorianFunke', 'Fundy', 'GeminiTay', 'GeorgeNotFound', 'GizzyGazza', 'Graser10', 'Grian', 'HBomb94', 'Hbomb94', 'Illumina', 'InTheLittleWood', 'ItsJustOriah', 'JackManifoldTV', 'JackSucksAtLife', 'JamesCharles', 'JamesTurner', 'Jameskii', 'JcTheCaster', 'JeromeASF', 'Jestannii', 'JoeyGraceffa', 'KaraCorvus', 'KarlJacobs', 'Katherineelizabeth', 'KingBurren', 'Kontuz', 'Krinios', 'Krtzyy', 'KryticZeuZ', 'LDShadowLady', 'Laurenzside', 'Ludwig', 'Marielitai', 'Mefs', 'Michaelmcchill', 'MiniMuka', 'NettyPlays', 'Nihachu', 'OrionSound', 'PEARLBYTEZ', 'PearlescentMoon', 'PeteZahHutt', 'Ph1LzA', 'Plumbella', 'Pokimane', 'Ponk', 'PrestonPlayz', 'Punz', 'Quackity', 'Quig', 'RTGame', 'Rafessor', 'Ranboo', 'Rendog', 'RipMika', 'Roguskii', 'Ryguyrocky', 'SB737', 'Sapnap', 'ScotGriswold', 'Seapeekay', 'Shubble', 'Skeppy', 'Smajor1995', 'Smallishbeans', 'Sneegsnag', 'SolidarityGaming', 'Spifey', 'Steph0sims', 'Strawburry17', 'Sylvee', 'TankMatt', 'TapL', 'Technoblade', 'TommyInnit', 'ToxxxicSupport', 'Tubbo', 'Vikkstar', 'Vixella', 'Voiceoverpete', 'WilburSoot', 'Wisp', 'Wolv21', 'Yammy', 'YeetDaisie', 'fWhip', 'fruitberries', 'iHasCupquake', 'iJevin', 'iicbunny', 'iskall85', 'uwuCorbin', 'vGumiho']

@app.route('/')
def index():
	return render_template("index.html", data="Your team will place around position ___")

@app.route("/prediction", methods=["POST"])



def prediction():
	team_names = [x for x in request.form.values()]
	print(team_names)

	# Preprocess Image
	predictionList = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
	index1 = colList.index(team_names[0])
	index2 = colList.index(team_names[1])
	index3 = colList.index(team_names[2])
	index4 = colList.index(team_names[3])

	predictionList[0][index1] = 1
	predictionList[0][index2] = 1
	predictionList[0][index3] = 1
	predictionList[0][index4] = 1

	predictionDF = pd.DataFrame(predictionList)

	# predictions
	pred = loaded_model.predict(predictionDF)
	pred_final = round(pred[0][0])
	output = 'Team: ' + team_names[0] + ', ' + team_names[1] + ', ' + team_names[2] + ', ' + team_names[3] + ' will place around position #' + str(pred_final)
	print(output)

	return render_template("index.html", data=output)


if __name__ == "__main__":
	app.run(debug=True)