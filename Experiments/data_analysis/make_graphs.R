require(Rmisc)
library(ggplot2)

allResults <- experimental_results
allResults$Seconds <- floor(allResults$Seconds)


small60k <- subset(allResults[order(allResults$ProcessorType, allResults$Nodes),], DataSize == "Small MNIST" & Model == "Small CNN" & Nodes > 1)
small600k <- subset(allResults[order(allResults$ProcessorType, allResults$Nodes),], DataSize == "Med MNIST" & Model == "Small CNN" & Nodes > 1)
small6000k <- subset(allResults[order(allResults$ProcessorType, allResults$Nodes),], DataSize == "Big MNIST" & Model == "Small CNN" & Nodes > 1)

med60k <- subset(allResults[order(allResults$ProcessorType, allResults$Nodes),], DataSize == "Small MNIST" & Model == "Med CNN" & Nodes > 1)
med600k <- subset(allResults[order(allResults$ProcessorType, allResults$Nodes),], DataSize == "Med MNIST" & Model == "Med CNN" & Nodes > 1)
med6000k <- subset(allResults[order(allResults$ProcessorType, allResults$Nodes),], DataSize == "Big MNIST" & Model == "Med CNN" & Nodes > 1)

big60k <- subset(allResults[order(allResults$ProcessorType, allResults$Nodes),], DataSize == "Small MNIST" & Model == "Big CNN" & Nodes > 1)
big600k <- subset(allResults[order(allResults$ProcessorType, allResults$Nodes),], DataSize == "Med MNIST" & Model == "Big CNN" & Nodes > 1)
big6000k <- subset(allResults[order(allResults$ProcessorType, allResults$Nodes),], DataSize == "Big MNIST" & Model == "Big CNN" & Nodes > 1)


calculateSpeedupOverCPU <- function(df){
  return(data.frame("Speedup" = (subset(df, ProcessorType == "CPU")$Seconds / subset(df, ProcessorType == "Graphics")$Seconds), 
                    "Nodes" = c(2,4,8,16),
                    "DataSize" = subset(df, ProcessorType == "CPU")$DataSize,
                    "Model" = subset(df, ProcessorType == "CPU")$Model))
}


small60kSpeedup <- calculateSpeedupOverCPU(small60k)
small600kSpeedup <- calculateSpeedupOverCPU(small600k)
small6000kSpeedup <- calculateSpeedupOverCPU(small6000k)

med60kSpeedup <- calculateSpeedupOverCPU(med60k)
med600kSpeedup <- calculateSpeedupOverCPU(med600k)
med6000kSpeedup <- calculateSpeedupOverCPU(med6000k)

big60kSpeedup <- calculateSpeedupOverCPU(big60k)
big600kSpeedup <- calculateSpeedupOverCPU(big600k)
big6000kSpeedup <- calculateSpeedupOverCPU(big6000k)

allSpeedup <- do.call("rbind", list(small60kSpeedup,small600kSpeedup,small6000kSpeedup,med60kSpeedup,med600kSpeedup,med6000kSpeedup,big60kSpeedup,big600kSpeedup,big6000kSpeedup))
allSpeedup

allSpeedupAvg <- aggregate(allSpeedup[, 1], list(allSpeedup$DataSize, allSpeedup$Model), mean)


tgc2 <- summarySE(small60k, measurevar='Seconds', groupvars=c("ProcessorType"))
tgc2




#makeDifGraph <-function(df){
temp <- allSpeedupAvg


#temp[,1] = factor(temp[,1], labels = c("60k", "600K", "6000k"))
temp[,1] = factor(temp[,1], labels = c("5MB", "500MB", "5GB"))
temp[,2] = factor(temp[,2], labels = c("34K", "6M", "69M"))


speedupBarGraph <- ggplot(temp, aes(temp[,2], x, fill = temp[,1]))+
  labs(fill = "Data Size")+
  geom_bar(stat="identity",position=position_dodge())+
  #scale_x_continuous(trans='log2')+
  #scale_y_continuous(trans='sqrt')+
  geom_text(aes(label=signif(x, digits = 3)),
            vjust=1.5, 
            color="white",
            fontface = "bold",
            position = position_dodge(0.9), 
            size=4.5)+
  scale_color_manual(labels = c('test','test','test'))+
  ggtitle('Speedup Comparisons', subtitle="Average Speedup Acrross All Node Counts") +
  #scale_fill_manual(values=c('#0071c5','#76b900', '#76b900'))+
  xlab('NN Model Size*') +
  ylab('Speedup Factor')+
  theme(axis.title.y = element_text(angle = 90)) +
  theme(axis.title.y = element_text(vjust = 0.5)) +
  labs(color = 'Thor Processor Type') +
  theme(legend.title.align=0.5) + 
  theme(
    panel.background = element_rect(fill = 'white', colour = 'black'),
    panel.grid.major = element_line(colour = "grey70"),
    panel.grid.minor = element_line(colour = "grey90"),
    axis.title=element_text(size=24),
    plot.title=element_text(size=24),
    plot.subtitle=element_text(size=20),
    axis.text=element_text(size=18)
  ) + 
  theme(plot.margin=unit(c(.25,.25,.25,.25),"in"))+
  labs(caption = "* number of trainable parameters")
  
  

makeLineGraph <- function(df){

lineData <- df
graph <- ggplot(lineData, aes(Nodes, Seconds, colour = ProcessorType)) + 
  geom_point()+
  geom_line()+
  scale_x_continuous(trans='sqrt', breaks = c(1,2,4,8,16)) +
  scale_y_continuous(trans='sqrt', breaks = lineData$Seconds)+
  #ggtitle('Fig. 3.3b | Training Time Decrease', subtitle='Time vs. # of Thor Nodes') + 
  ggtitle('GPU vs CPU', subtitle=df$Name) + 
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(plot.subtitle = element_text(hjust = 0.5)) +
  xlab('Thor Nodes') +
  ylab('Training Time\n(seconds)') +
  theme(axis.title.y = element_text(angle = 90)) +
  theme(axis.title.y = element_text(vjust = 0.5)) +
  labs(color = 'Thor Processor Type') +
  theme(legend.title.align=0.5) + 
  theme(
    panel.background = element_rect(fill = 'white', colour = 'black'),
    panel.grid.major = element_line(colour = "grey70"),
    panel.grid.minor = element_line(colour = "grey90"),
    axis.title=element_text(size=24),
    plot.title=element_text(size=24),
    plot.subtitle=element_text(size=20),
    axis.text=element_text(size=18)
  ) + 
  theme(plot.margin=unit(c(.25,.25,.25,.25),"in"))

return(graph)
}

makeBarGraph <- function(df){

barData <- df
graph <- ggplot(data=barData, aes(x=Nodes, y=Seconds, fill=ProcessorType)) +
  geom_bar(stat="identity",position=position_dodge())+
  scale_x_continuous(trans='log2')+
  #scale_y_continuous(trans='sqrt')+
  geom_text(aes(label=Seconds), vjust=1.2, color="white",position = position_dodge(0.9), size=3.5)+
  ggtitle('GPU vs CPU', subtitle=df$Name) +
  scale_fill_manual(values=c('#0071c5','#76b900'))+
  xlab('Thor Nodes') +
  ylab('Training Time\n(seconds)')

return(graph)
}


#Line Graphs
ggsave('graphs/small60k.png', plot=makeLineGraph(small60k), width=10, height=7.5, dpi=1048)
ggsave('graphs/small600k.png', plot=makeLineGraph(small600k), width=10, height=7.5, dpi=1048)
ggsave('graphs/small6000k.png', plot=makeLineGraph(small6000k), width=10, height=7.5, dpi=1048)

ggsave('graphs/med60k.png', plot=makeLineGraph(med60k), width=10, height=7.5, dpi=1048)
ggsave('graphs/med600k.png', plot=makeLineGraph(med600k), width=10, height=7.5, dpi=1048)
ggsave('graphs/med6000k.png', plot=makeLineGraph(med6000k), width=10, height=7.5, dpi=1048)

ggsave('graphs/big60k.png', plot=makeLineGraph(big60k), width=10, height=7.5, dpi=1048)
ggsave('graphs/big600k.png', plot=makeLineGraph(big600k), width=10, height=7.5, dpi=1048)
ggsave('graphs/big6000k.png', plot=makeLineGraph(big6000k), width=10, height=7.5, dpi=1048)

#Bar Graphs
ggsave('graphs/speedup.png', plot=speedupBarGraph, width=10, height=7.5, dpi=1048)

ggsave('graphs/small60kbar.png', plot=makeBarGraph(small60k), width=10, height=7.5, dpi=1048)
ggsave('graphs/small600kbar.png', plot=makeBarGraph(small600k), width=10, height=7.5, dpi=1048)
ggsave('graphs/small6000kbar.png', plot=makeBarGraph(small6000k), width=10, height=7.5, dpi=1048)

ggsave('graphs/med60kbar.png', plot=makeBarGraph(med60k), width=10, height=7.5, dpi=1048)
ggsave('graphs/med600kbar.png', plot=makeBarGraph(med600k), width=10, height=7.5, dpi=1048)
ggsave('graphs/med6000kbar.png', plot=makeBarGraph(med6000k), width=10, height=7.5, dpi=1048)

ggsave('graphs/big60kbar.png', plot=makeBarGraph(big60k), width=10, height=7.5, dpi=1048)
ggsave('graphs/big600kbar.png', plot=makeBarGraph(big600k), width=10, height=7.5, dpi=1048)
ggsave('graphs/big6000kbar.png', plot=makeBarGraph(big6000k), width=10, height=7.5, dpi=1048)


