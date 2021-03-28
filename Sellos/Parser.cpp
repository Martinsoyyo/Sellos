#include "pch.h"
#include "Parser.h"

void Remove(string& STR, const string& PHRASE) {
	STR = STR.substr(STR.find(PHRASE) + string(PHRASE).length() + 1);
}

std::vector<size_t> GetParams(const string& STR) {
	std::istringstream is(STR);
	return(std::vector<size_t>((std::istream_iterator<size_t>(is)), (std::istream_iterator<size_t>())));
}

Parser::Parser(const string& TEST) :m_test_name(TEST)
{
	std::ifstream inputFileStream(FILENAME);
	std::string str((std::istreambuf_iterator<char>(inputFileStream)), (std::istreambuf_iterator<char>()));

	Parse_General_Configuration(str);
	Parse_Net_Configuration(str);
};

void Parser::Parse_General_Configuration(string& STR)
{
	Remove(STR, "[GENERAL CONFIGURATION]");
	Remove(STR, "[EPOCH]"); m_epoch = stoi(STR);
	Remove(STR, "[BATCH SIZE]"); m_batch_size = stoi(STR);
	Remove(STR, "[ROOT FOLDER]"); m_root_folder = STR.substr(0, STR.find("\n"));
	Remove(STR, "[INPUT CHANNELS]"); m_input_channel = stoi(STR);
	Remove(STR, "[IMAGE SIZE]"); m_image_size = stoi(STR);
	Remove(STR, "[OUTPUT CHANNELS]"); m_output_channel = stoi(STR);
	Remove(STR, "[PERCENT TO TRAIN]"); m_percent_to_train = stof(STR);

};

void Parser::Parse_Net_Configuration(string& STR)
{
	Remove(STR, m_test_name);
	Remove(STR, "[MODEL TYPE]"); m_model_type = STR.substr(0, STR.find("\n"));
	Remove(STR, "[CONV LAYER CONFIGURATION]"); m_conv_layer_conf = GetParams(STR);
	Remove(STR, "[LINEAL LAYER CONFIGURATION]"); m_linear_layer_conf = GetParams(STR);
	Remove(STR, "[BATCH NORM]"); m_batch_norm = stoi(STR);
	Remove(STR, "[DROP OUT]"); m_drop_out = stof(STR);
};