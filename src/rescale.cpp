#include <iostream>
#include <fstream>
#include <vector>
#include <string>

int main()
{
    struct data
    {
        double k;
        double p_s;
    };

    std::ifstream in("power_spectrum.dat");
    std::vector<data> row;
    data r;
    const double factor = 0.05;   // k_star in Mpc^-1

    std::string header;
    std::getline(in, header);     

    while (in >> r.k >> r.p_s)
    {
        row.push_back(r);
    }

    std::ofstream out("power_spectrum_Mpc.dat");
    out << "# k [Mpc^-1]  P_s\n";
    for (auto i : row)
    {
        out << i.k * factor << " " << i.p_s << '\n';
    }

    std::cout << "Done" << '\n';
    return 0;
} 

  
