#pragma once
#include "pch.h"

#define FILENAME "CONFIG.txt"

class Parser {
public:
	Parser(const string& TEST);

	ifstream m_file;
	string m_test_name;
	size_t m_epoch;
	size_t m_batch_size;
	string m_root_folder;
	size_t m_input_channel;
	size_t m_output_channel;
	size_t m_image_size;
	float  m_percent_to_train;

	string m_model_type;
	vector<string> m_conv_layer_conf;
	vector<string> m_linear_layer_conf;
	bool m_batch_norm;
	float m_drop_out;
};
