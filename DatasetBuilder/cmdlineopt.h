#pragma once

namespace CmdLineOpt {

	void CmdLineOpt(int argc, const char* argv[]);

	extern bool verbose;
	extern int image_size;
	extern std::string dataset_path;
	extern std::string dataset_prefix;
};