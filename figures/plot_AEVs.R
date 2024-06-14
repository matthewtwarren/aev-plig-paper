library(tidyverse)
library(data.table)
library(RColorBrewer)
library(gridExtra)

######################################
## Plot cutoff and contribution fns ##
######################################

cutoff_fn_ANI <- function(x, Rc) {
  return(0.5*cos(pi*(x/Rc)) + 0.5)
}

radial_ANI <- function(x, Rc, Eta, Rs) {
  return(exp(-Eta*(x - Rs)^2)*cutoff_fn_ANI(x, Rc))
}


RcR = 5.1 # Radial cutoff
EtaR = 19.7 # Radial decay
RsR_vec = c(0.80, 1.34, 1.88, 2.41,
            2.95, 3.49, 4.03, 4.56) # Radial shift


R <- seq(0, RcR, 0.01)
Data <- data.frame(R)
for(RsR in RsR_vec) {
  newcolname <- paste0("Rs = ", RsR) 
  Data[[newcolname]] <- radial_ANI(R, RcR, EtaR, RsR)
}

data_long <- melt(as.data.table(Data), id.vars = 'R')
colnames(data_long) <- c('R', 'Center', 'value')
custom.col <- c("#FFDB6D", "#C4961A", "#F4EDCA", 
                "#D16103", "#C3D7A4", "#52854C", "#4E84C4", "#293352")

ggplot(data_long, aes(x = R, y = value, color = Center)) +
  geom_line(size = 0.7) +
  scale_color_brewer(palette = "Dark2") +
  ylab(expression('C(R'['ij']*')')) +
  xlab(expression('R'['ij']*'  [Å]')) +
  labs(color = "Center  [Å]\n") +
  theme_grey(base_size = 16)

expression('exp[-'*eta*'(R'['ij']*' - '*'R'['s']*')'^2*']f'['c']*'(R'['ij']*')')
plot(x, radial_ANI(x, RcR, EtaR, RsR_vec[1]), type = 'l', col = 'blue')
