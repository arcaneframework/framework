using System;
using System.Diagnostics;
using System.IO;
using Arcane.ExecDrivers.Common;

namespace Arcane.ExecDrivers.Common
{
  class CustomMpiDriver : ICustomExecDriver
  {
    bool m_is_init;

    public string Name { get { return "CustomMpiDriver"; } }

    void _Init()
    {
      if (m_is_init)
        return;
      string host_name = Environment.MachineName;
      Console.WriteLine("TRY CUSTOM MPI_LAUNCHER machine={0}");
      m_is_init = true;
    }

    public bool HandleMpiLauncher(Arcane.ExecDrivers.Common.ExecDriverProperties p, string mpi_launcher)
    {
      _Init();
      Console.WriteLine("TRY CUSTOM MPI_LAUNCHER name={0}", mpi_launcher);
      Console.WriteLine("OS_VERSION={0}", Environment.OSVersion.VersionString);
      if (mpi_launcher.EndsWith("ccc_mprun", StringComparison.Ordinal))
        return _HandleCCCMprun(p);
      if (mpi_launcher.EndsWith("mpiexec", StringComparison.Ordinal))
        return _HandleMpiExec(p);
      return false;
    }

    bool _HandleMpiExec(ExecDriverProperties p)
    {
      Console.WriteLine("Handle Mpiexec");
      return false;
    }

    // Gère le lancement via 'ccc_mprun'
    bool _HandleCCCMprun(ExecDriverProperties p)
    {
      Console.WriteLine("Handling 'ccc_mprun'");

      // Si on est dans un job slurm, on suppose qu'on est sur un noeud de l'allocation
      // (ce n'est pas forcément le cas. Pour être sur, il faudrait regarder si le noeud
      // sur lequel on tourne est dans la liste des noeuds de la variable d'environnement
      // SLURM_JOB_NODELIST.
      string job_id = Utils.GetEnvironmentVariable("SLURM_JOB_ID");
      bool is_already_allocated = !string.IsNullOrEmpty(job_id);
      Console.WriteLine("is_already_allocated={0}", is_already_allocated);

      if (is_already_allocated) {
        // Avec ccc_mprun, il faut ajouter au srun l'option '--exclusive' pour ne pas
        // avoir de problèmes de cohérence sur le nombre de processus à utiliser.
        p.MpiLauncherArgs.Add("-E");
        p.MpiLauncherArgs.Add("--exclusive");
      }

      if (p.UseTotalview) {
        p.MpiLauncherArgs.Add("-d tv");
        p.UseTotalview = false;
      }
      int nb_task = Math.Max(p.NbTaskPerProcess,1);
      int nb_shared_memory = Math.Max(p.NbSharedMemorySubDomain,1);
      int nb_thread = nb_task * nb_shared_memory;
      Console.WriteLine("NB_THREAD={0} (nb_shm={1},nb_task={2})", nb_thread, nb_shared_memory, nb_task);

      p.MpiLauncherArgs.Add("-n");
      p.MpiLauncherArgs.Add(p.NbProc.ToString());

      if (!is_already_allocated) {
        p.MpiLauncherArgs.AddRange(new string[] { "-x", "-v", });
      }
      // Alloue les coeurs suivant le nombre de threads demandes
      // et specifie un nombre de noeuds pour etre sur que les threads seront bien repartis
      if (nb_thread > 1) {
        int nb_thread_power_2 = 1;
        while (nb_thread_power_2 < nb_thread)
          nb_thread_power_2 *= 2;
        p.MpiLauncherArgs.Add("-c" + nb_thread_power_2);
        //Utils.SetEnvironmentVariable("ARCANE_BIND_THREADS", "1");
        //Utils.SetEnvironmentVariable("ARCANE_SPINLOCK_BARRIER", "1");
      }

      return true;
    }
  }
}
