#include <iostream>
#include <torch/torch.h>

int main() {
    torch::Tensor t1 = torch::rand({2, 3});
    std::cout << t1 << std::endl;
    return 0;
}
