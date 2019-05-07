import pylab
from pycbc.waveform import get_td_waveform

for mass_1 in range(10, 100, 1):
    for mass_2 in range(10, 100, 1):
        hp, hc = get_td_waveform(approximant='IMRPhenomC',
                                     mass1=mass_1,
                                     mass2=mass_2,
                                     spin1z=0,
                                     delta_t=1.0/4096,
                                     f_lower=40)

        f = open('CBC Waveforms/' + str(mass_1) + '_' + str(mass_2) + '.dat', 'w+')

        for j in range(len(hp.sample_times)):
            f.write(str(hp.sample_times[j]) + ' ' + str(hp[j]) + '\n')

        f.close()

        # pylab.plot(hp.sample_times, hp, label=str(mass_1))


# pylab.ylabel('Strain')
# pylab.xlabel('Time (s)')
# pylab.legend()
# pylab.show()
