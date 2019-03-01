#include <iostream>
#include <torch/script.h>

int main(int argc, const char* argv[]){
	if (argc != 2){
		std::cerr << "where is your model?" << std::endl;
		return -1;
	}
	std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(argv[1]);
	assert(module != nullptr);
	std::cout << "load successfully!" << std::endl;

	//auto module = torch::jit::load("../resnet.pt");
	std::vector<torch::jit::IValue> inputs;
	inputs.push_back(torch::ones({1,3,224,224}));
	at::Tensor output = module->forward(inputs).toTensor();
	std::cout << output.slice(1,0,5);
	return 0;
}
