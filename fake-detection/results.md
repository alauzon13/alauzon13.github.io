
In 2019, The RealReal \-- an “authenticated luxury resale” marketplace \-- sold a fake Dior bag to [Forbes contributor](https://www.forbes.com/sites/richardkestenbaum/2019/10/23/if-fake-bags-are-being-sold-on-the-realreal-how-can-the-resale-business-ever-succeed/) Richard Kestenbaum for $3,600 USD. The inauthenticity was determined by external experts who were trained to spot the tells: the herringbone pattern was slightly off; the handles were a hair too short; the embroidery wasn’t as detailed as it should be. 

When contacted about the legitimacy of the bag, The RealReal said they were not able to definitively confirm, stating that authenticating is both “an art and a science.” Further, when asked about the quality of TRR’s internal authentication process, inside sources said that authenticators did “as good a job as any human can.” 

Being both a data scientist and a fashion lover, these comments stuck with me. Kestenbaum also points out that the average customer doesn’t have access to trained authenticity experts. If there is a scientific element to authenticating, if human error is a concern, and if there is an accessibility gap in authentication services, why not turn to machine learning? With the right data and models, an algorithmic approach could make luxury authentication more consistent, scalable, and accessible. 

### The Big Idea 

As a proof-of-concept approach, I wanted to train a binary image classification model to identify real versus fake versions of a single designer bag. I chose to limit the scope to one style because signs might be bag-specific \-- like the embroidery in the fake Dior sold by The RealReal.

### The bag 

When it came to the bag, I knew immediately which one I wanted to investigate: the Hermès Birkin. The Birkin is a universal symbol of wealth and exclusivity, with prices starting at $10,000 USD. The elusivity of Birkins is maintained by the preconditions of purchase: you can’t just walk into a store and buy one. Instead, customers must first build credit with the brand by purchasing other items before they can be considered as potential buyers of a Birkin. They must develop a relationship with a designated sales associate, who plays a critical role in deciding who is able to purchase a bag. The road to a Birkin can be years long, resulting in some unhappy customers [suing](https://www.nytimes.com/2024/03/21/style/birkin-bag-hermes-lawsuit.html) the French luxury brand for violating antitrust law. 

I was particularly interested in the Hermès Himalaya Birkin, a white-and-grey crocodile version of the famous bag. The Himalaya is regarded as one of the most elusive and expensive handbags in the world, with a diamond-encrusted version being sold for over $450,000 USD in 2022 (the most expensive bag sale to date). Even without diamonds, the white and grey crocodile skin bag sells for about $200,000 new. 

![himalaya-birk](assets/himalaya-birk.webp)

It’s no wonder that, given what a Himalaya communicates about one's time and income, that they are one of the most-duped bags available on the replica market. Just this past December, a replica for sale on [Walmart](https://www.cbc.ca/news/world/walmart-wirkin-bag-1.7423682) went viral and subsequently sold out, and other versions are readily available on sites like DHgate and AliExpress. 

### The data 

### The Model 

### The results

