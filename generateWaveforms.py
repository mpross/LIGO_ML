import pylab
from pycbc.waveform import get_td_waveform

for mass_1 in range(10, 30):
    for mass_2 in range(10, 30):
        hp, hc = get_td_waveform(approximant='IMRPhenomC',
                                     mass1=mass_1/10.0,
                                     mass2=mass_2/10.0,
                                     spin1z=0,
                                     delta_t=1.0/4096,
                                     f_lower=40)

        f = open('NS Waveforms/' + str(mass_1/10.0) + '_' + str(mass_2/10.0) + '.dat', 'w+')

        for j in range(len(hp.sample_times)-1*4096, len(hp.sample_times)):
            f.write(str(hp.sample_times[j]) + ' ' + str(hp[j]) + '\n')

        f.close()

        print('M1: '+str(mass_1/10.0)+' M_2: '+str(mass_2/10.0))

    # pylab.plot(hp.sample_times, hp, label=str(mass_1))

# pylab.ylabel('Strain')
# pylab.xlabel('Time (s)')
# pylab.legend()
# pylab.show()
