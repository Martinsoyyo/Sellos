#include "pch.h"
#include "Parser.h"


bool Remove(string& STR, const string& PHRASE) {
	auto IDX = STR.find(PHRASE);
	if (IDX != string::npos) {
		STR = STR.substr(STR.find(PHRASE) + string(PHRASE).length()+1);
		return true;
	}
	return false;
}


vector<string> GetParams(const string& STR) {
	std::istringstream ss(STR);
	std::istream_iterator<std::string> begin(ss);
	std::istream_iterator<std::string> end;

	return(vector<string>(begin, end));
}


Parser::Parser(const string& TEST) :m_test_name(TEST)
{
	std::ifstream inputFileStream(FILENAME);
	string STR;

	while (getline(inputFileStream, STR))
	{
		//PARAMETROS GENERALES DE ENTRENAMIENTO
			 if (Remove(STR, "[EPOCH]"))			m_epoch = stoi(STR);
		else if (Remove(STR, "[BATCH_SIZE]"))		m_batch_size = stoi(STR);
		else if (Remove(STR, "[ROOT_FOLDER]"))		m_root_folder = STR.substr(0, STR.find("\n"));
		else if (Remove(STR, "[INPUT_CHANNELS]"))	m_input_channel = stoi(STR);
		else if (Remove(STR, "[IMAGE_SIZE]"))		m_image_size = stoi(STR);
		else if (Remove(STR, "[OUTPUT_CHANNELS]"))	m_output_channel = stoi(STR);
		else if (Remove(STR, "[PERCENT_TO_TRAIN]")) m_percent_to_train = stof(STR);

		else if (Remove(STR, m_test_name)) {
			getline(inputFileStream, STR);
			Remove(STR, "[MODEL_TYPE]"); m_model_type = STR.substr(0, STR.find("\n")); 
			
			getline(inputFileStream, STR);
			Remove(STR, "[CONV_LAYER_CONFIGURATION]");  m_conv_layer_conf = GetParams(STR);

			getline(inputFileStream, STR);
			Remove(STR, "[LINEAL_LAYER_CONFIGURATION]");  m_linear_layer_conf = GetParams(STR);
			
			if (m_model_type == "VGG") {
				getline(inputFileStream, STR);
				Remove(STR, "[BATCH_NORM]");  m_batch_norm = stoi(STR);

				getline(inputFileStream, STR);
				Remove(STR, "[DROP_OUT]");  m_drop_out = stof(STR);
			}
			break;
		};
	};

};

