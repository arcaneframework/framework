
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>
#include <sstream>
#include <composyx.hpp>
#include <composyx/part_data/PartMatrix.hpp>

int main(int argc, char *argv[])
{
    using namespace composyx;
    using Scalar = double;

    MMPI::init();
    const int rank = MMPI::rank();
    std::stringstream filename;
    filename<<"output"<<rank<<".txt";
    std::ofstream fout(filename.str()) ;
    {
      const int N_DOFS = 8;
      const int N_SUBDOMAINS = 4;

      // Bind subdomains to MPI process
      std::shared_ptr<Process> p = bind_subdomains(N_SUBDOMAINS);

      // Define topology for each subdomain
      std::vector<Subdomain> sd;
      if(p->owns_subdomain(0)){
        std::map<int, std::vector<int>> nei_map_0 {{1, {1, 3, 4, 5}},
                                                   {2, {2, 3, 6, 7}}};
        std::map<int, std::vector<int>> nei_owner_map {{1, {0, 0, 1, 1}},
                                                   {2, {0, 0, 2, 2}}};
        sd.emplace_back(0, N_DOFS, std::move(nei_map_0),nei_owner_map,false);
      }
      if(p->owns_subdomain(1)){
        std::map<int, std::vector<int>> nei_map_1 {{0, {4, 5, 0, 2}},
                                                   {3, {2, 3, 6, 7}}};
        std::map<int, std::vector<int>> nei_owner_map {{0, {0, 0, 1, 1}},
                                                   {2, {1, 1, 2, 2}}};
        sd.emplace_back(1, N_DOFS, std::move(nei_map_1),nei_owner_map,false);
      }
      if(p->owns_subdomain(2)){
        std::map<int, std::vector<int>> nei_map_2 {{0, {4, 5, 0, 1}},
                                                   {3, {1, 3, 6, 7}}};
        std::map<int, std::vector<int>> nei_owner_map {{0, {0, 0, 2, 2}},
                                                   {3, {2, 2, 3, 3}}};
        sd.emplace_back(2, N_DOFS, std::move(nei_map_2),nei_owner_map,false);
      }
      if(p->owns_subdomain(3)){
        std::map<int, std::vector<int>> nei_map_3 {{1, {4, 5, 0, 1}},
                                                   {2, {6, 7, 0, 2}}};
        std::map<int, std::vector<int>> nei_owner_map {{1, {1, 1, 3, 3}},
                                                   {2, {2, 2, 3, 3}}};
        sd.emplace_back(3, N_DOFS, std::move(nei_map_3),nei_owner_map,false);
      }
      p->load_subdomains(sd);
      p->display("Process",fout) ;
    }
    MMPI::finalize();

    return 0;
}

