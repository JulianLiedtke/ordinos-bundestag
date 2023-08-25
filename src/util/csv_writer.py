import csv
import logging
import threading

log = logging.getLogger(__name__)

class CSV_Writer:
    """ This class is responsible for writing the measured times in CSV-files """
    csvfile = None
    writer = None
    preparation = 0
    evaluation = 0
    fieldnames = []

    @classmethod
    def set_prep_time(cls, time):
        """ set time for preparation of the system / adding votes """
        log.debug('Preparation and vote aggregation: {:.3f}s'.format(time))
        cls.preparation = time

    @classmethod
    def set_eval_time(cls, time):
        """ set time for computation of the election result"""
        log.debug('Evaluation time: {:.3f}s'.format(time))
        cls.evaluation = time

    @classmethod
    def init_writer(cls):
        # TODO: path of files
        cls.fieldnames = ['Candidates', 'Voter', 'Prep-Time(s)', 'Eval-Time(s)', 'Winners', 'gt-ops', 'eq-ops', 'dec-ops', 'mul-ops', 'System-Name']
        cls.csvfile = open('results/times.csv', 'w', newline='')
        writer = csv.DictWriter(cls.csvfile, fieldnames=cls.fieldnames)
        writer.writeheader()
        cls.csvfile.close()

    @classmethod
    def write_with_election_params(cls, num_cand, num_voter, n_const, rand_before, system_id, winners=None, gt_operations=None, eq_operations=None, dec_operations=None, mul_operations=None):
        """ write time-relevant parameters and times """
        if len(cls.fieldnames) == 0:
            log.warning("CSV_Writer is not initialized")
            return
        # TODO: path of files
        cls.csvfile = open('results/times.csv', 'a', newline='')
        writer = csv.DictWriter(cls.csvfile, fieldnames=cls.fieldnames)
        prep = '%.2f'%(cls.preparation)
        eval = '%.2f'%(cls.evaluation)

        writer.writerow({cls.fieldnames[0]: num_cand,
                        cls.fieldnames[1]: num_voter,
                        cls.fieldnames[2]: prep,
                        cls.fieldnames[3]: eval,
                        cls.fieldnames[4]: winners,
                        cls.fieldnames[5]: n_const,
                        cls.fieldnames[6]: rand_before,
                        #cls.fieldnames[5]: gt_operations,
                        #cls.fieldnames[6]: eq_operations,
                        cls.fieldnames[7]: dec_operations,
                        cls.fieldnames[8]: mul_operations,
                        cls.fieldnames[9]: system_id})
        
        cls.csvfile.close()

        log.debug("Succesfully written preparation/aggregation time: " + str(prep))
        log.debug("Succesfully written evaluation time: " + str(eval))

    @classmethod
    def write_times(cls, step, time):
        if len(cls.fieldnames) == 0:
            log.warning("CSV_Writer is not initialized")
            return
        cls.csvfile2 = open('results/real_world_times.csv', 'a', newline='')
        writer2 = csv.DictWriter(cls.csvfile2, fieldnames=cls.fieldnames)
       # prep = '%.2f'%(cls.preparation)
       # eval = '%.2f'%(cls.evaluation)

        writer2.writerow({cls.fieldnames[0]: step,
                        cls.fieldnames[1]: time,
                        })
        

    @classmethod
    def write_times_benchmarks(cls, id, step, time):
        if len(cls.fieldnames) == 0:
            log.warning("CSV_Writer is not initialized")
            return
        cls.csvfile2 = open('results/times/'+str(id)+'.csv', 'a', newline='')
        writer2 = csv.DictWriter(cls.csvfile2, fieldnames=cls.fieldnames)
 

        writer2.writerow({cls.fieldnames[0]: step,
                        cls.fieldnames[1]: time,
                        })
        
        cls.csvfile2.close()

    @classmethod
    def write_result_benchmarks(cls, id, step, result):
        if len(cls.fieldnames) == 0:
            log.warning("CSV_Writer is not initialized")
            return
        cls.csvfile2 = open('results/results/'+str(id)+'.csv', 'a', newline='')
        writer2 = csv.DictWriter(cls.csvfile2, fieldnames=cls.fieldnames)
 

        writer2.writerow({cls.fieldnames[0]: step,
                        cls.fieldnames[1]: result,
                        })
        
        cls.csvfile2.close()

    @classmethod
    def write_results(cls, step, results):
        if len(cls.fieldnames) == 0:
            log.warning("CSV_Writer is not initialized")
            return
        cls.csvfile2 = open('results/results.csv', 'a', newline='')
        writer2 = csv.DictWriter(cls.csvfile2, fieldnames=cls.fieldnames)

        writer2.writerow({cls.fieldnames[0]: step,
                        cls.fieldnames[1]: results,
                        })
        cls.csvfile2 = open('results/results.csv', 'a', newline='')

    

    @classmethod
    def write_error(cls, info, comment):
        """ write the exception info if an exception occured """
        if len(cls.fieldnames) == 0:
            log.warning("CSV_Writer is not initialized")
            return
        # TODO: path of files
        cls.csvfile = open('results/times.csv', 'a', newline='')
        writer = csv.DictWriter(cls.csvfile, fieldnames=cls.fieldnames)

        writer.writerow({cls.fieldnames[0]: info[0], cls.fieldnames[1]: info[1], cls.fieldnames[2]: comment})
        cls.csvfile.close()
